#!/bin/bash

# To test a fresh install:
# ./aws_op.py -m test01 create
# ssh -i twoxar-platform-ec2.pem -L8000:localhost:8000 ubuntu@ip_goes_here
# sudo apt-get install git
# mkdir 2xar
# cd 2xar/
# git clone https://github.com/andrewradin/twoxar-demo.git
# cd twoxar-demo/
# ./install/install.sh

# root of git repository
repos=~/2xar/twoxar-demo

username=`whoami`

venv_py=~/2xar/opt/conda/envs/py3web1/bin/python3

mode=`$repos/web1/path_helper.py machine_type`
if [ $? != 0 ]; then
	echo "couldn't get machine_type from local_settings.py"
	exit 1
fi
# optional operations
update=true
venv=false
djangodb=true
newdb=false
apache=false
verbose=true
shutdown=false
checkdisk=false
machinestats=true
backup=false
check_lts=false
update_resources=true
dump_only=false
rnaseq=false
nltk_full=true
nltk_shared=false
timezone=true
drivebackground_user=$username

PGM=`basename $0`
while [ "$#" -gt 0 ]
do
	case $1 in
		-h|-\?|--help)
			echo "$PGM [options]"
			echo "options are:"
			echo "-h|-?|--help - print this message"
			echo "-u|--no-update - skip apt-get update"
			echo "-d|--dump-only - show selected activities and exit"
			exit
			;;
		-d|--dump-only)
			dump_only=true
			;;
		-u|--no-update)
			update=false
			;;
		*)
			echo "unknown option: $1; use -h for help"
			exit 1
			;;
	esac
	shift
done

upgrade=`$repos/web1/path_helper.py upgrade_on_install | tr '[A-Z]' '[a-z]'`
if [ "$upgrade" != 'true' -a "$upgrade" != 'false' ]
then
	echo "invalid upgrade_on_install config: $upgrade"
	exit 1
fi
check_aws=`$repos/web1/path_helper.py monitor_AWS | tr '[A-Z]' '[a-z]'`
if [ "$check_aws" != 'true' -a "$check_aws" != 'false' ]
then
	echo "invalid monitor_AWS config: $check_aws"
	exit 1
fi

if [ $mode == 'platform' ]
then
	apache=true
	if $repos/web1/path_helper.py is_production
	then
		backup=true
		check_lts=true
	fi
	nltk_full=false
	nltk_shared=true
	checkdisk=true
	checkdisk_user="www-data"
	drivebackground_user="www-data"
elif [ $mode == 'worker' ]
then
	djangodb=false
	shutdown=true
	checkdisk=true
	rnaseq=true
	nltk_full=false
    update_resources=false
	checkdisk_user="ubuntu"
	drivebackground_user=""
elif [ $mode == 'dev' ]
then
	: # keep defaults
elif [ $mode == 'dev-idle' ]
then
	: # keep defaults
    shutdown=true
else
	echo "invalid machine_type: $mode"
	exit 1
fi

if $dump_only
then
	set | grep -v '[A-Z]' | grep -v '^_'
	exit
fi

dbname=web1
banner="***********"

. /etc/lsb-release

$verbose && echo "$banner Installing apt packages..."

export DEBIAN_FRONTEND=noninteractive

$update && sudo apt-get -y update

# install needed packages
apt_pkgs=" \
	mysql-server \
	libmysqlclient-dev \
	git \
	git-annex \
	python-dev \
	python-virtualenv \
	libxml2-dev \
	libxslt1-dev \
	default-jre \
	curl \
	libcurl4-gnutls-dev \
	libcairo2-dev \
	libxt-dev \
	parallel \
	libglu1-mesa-dev \
	libssl-dev \
	flex \
	bison \
	build-essential \
	cmake \
	sqlite3 \
	libsqlite3-dev \
	libboost-dev \
	libboost-python-dev \
	libboost-regex-dev \
	libbz2-dev \
	liblzma-dev \
	mercurial \
	libgeos-dev \
	libgmp3-dev \
	libmpfr-dev \
	xvfb \
	libgconf2-4 \
	dos2unix \
	libffi-dev \
	poppler-utils \
	acl \
	libarchive-zip-perl \
	libtry-tiny-perl \
	libdbi-perl \
	tabix \
    libopenbabel-dev \
    openbabel \
    swig \
    unzip \
    libxss1 \
    firefox \
	rsyslog \
	cpio \
	wget \
	expect \
	" # end of list

if [ "$DISTRIB_RELEASE" == "20.04" ]
then
	apt_pkgs=${apt_pkgs/python-dev/python3-dev}
	apt_pkgs=${apt_pkgs/python-virtualenv/python3-virtualenv}
	apt_pkgs=${apt_pkgs/libgconf2-4/}
	apt_pkgs="${apt_pkgs} python-is-python3"
fi

# (Use -E to pass the debconf noninteractive env var)
sudo -E apt-get -y install $apt_pkgs || exit 1

# optionally upgrade packages
$update && $upgrade && sudo -E apt-get -y upgrade

# Remove any recently deprecated packages or directories. Since this only
# needs to happen once per machine, these lists can be cleared out periodically.
deprecated_pkgs=""
deprecated_dirs=""

if [ ! -z "$deprecated_pkgs" ]
then
	$verbose && echo "$banner Removing deprecated packages..."
	sudo apt-get -y remove $deprecated_pkgs
fi
if [ ! -z "$deprecated_dirs" ]
then
	$verbose && echo "$banner Removing deprecated directories..."
	sudo rm -rf $deprecated_dirs
fi

$verbose && echo "$banner Setting up syslog..."

INSTALL_DATE=`date +%Y%m%d`
rotate_log() {
	file="$1"
	dir="$2"
	from=$dir/$file
	to=$dir/$file-$INSTALL_DATE.gz
	if [ -e $from -a \! -e $to ]
	then
		sudo gzip ${from}
		sudo mv ${from}.gz ${to}
	fi
}

rotate_log django.log /var/log
sudo cp $repos/install/30-django.conf /etc/rsyslog.d
sudo service rsyslog restart

if [[ "$drivebackground_user" != "" ]]
then
	rotate_log drive_background.log /var/log
	sudo touch /var/log/drive_background.log
	sudo chown $drivebackground_user /var/log/drive_background.log
fi

if $timezone; then
    echo "Setting timezone to pacific"
    sudo timedatectl set-timezone "America/Los_Angeles"
fi



$verbose && echo "$banner Installing py3 packages..."

# Setup a python3 virtualenv.
if ! $(dirname "$0")/sub_install_py3.sh; then
    echo "Failed to setup python3"
    exit 1
fi
source $HOME/2xar/opt/conda/bin/activate py3web1

$verbose && echo "$banner Installing R packages..."

# Setup an R virtualenv.
if ! $(dirname "$0")/sub_install_R.sh; then
    echo "Failed to setup R"
    exit 1
fi

if ! $(dirname "$0")/sub_install_reactome.sh; then
    echo "Failed to setup reactome"
    exit 1
fi


if ! $(dirname "$0")/sub_install_js.sh; then
    echo "Failed to setup javascript"
    exit 1
fi

# Install geckodriver for testing purposes.
# (apt also has one - could we just use that?)
pushd $repos/selenium
make check_geckodriver_version
make geckodriver
popd

# check for NLTK data
# This is used by ae_search.
if $nltk_full && [ ! -d $HOME/nltk_data/ ]; then
       $venv_py -m nltk.downloader all
fi
if $nltk_shared; then
       nltk_dest=/usr/local/share/nltk_data
       if [ ! -d $nltk_dest ]; then
               sudo mkdir -p $nltk_dest
               sudo chown $username $nltk_dest
       fi
       for PKG in stopwords wordnet; do
               if [ ! -d $nltk_dest/corpora/$PKG ]; then
                       $venv_py -m nltk.downloader -d $nltk_dest $PKG
                       rm $nltk_dest/corpora/$PKG.zip
               fi
       done
fi


# build credits file
(cd $repos/web1/credits && ./build_credits_file.py) || exit 1

cd ~/2xar

$verbose && echo "$banner Installing weka..."
WEKA=weka-3-6-11
if [ ! -d $WEKA ]
then
	s3cmd get s3://duma-datasets/$WEKA.zip || exit 1
	unzip $WEKA.zip
	rm -f $WEKA.zip
fi

$verbose && echo "$banner Removing stale ws files..."
$repos/web1/path_helper.py --clean-stale || exit 1

$verbose && echo "$banner Installing legacy ws files..."
ws_s3_files="\
	AE_CC_search.model \
	AE_CC_search.pickle \
	AE_TR_search.model \
	AE_TR_search.pickle \
	" # end of list

for FILE in $ws_s3_files
do
	if [ ! -f ~/2xar/ws/$FILE ]
	then
		s3cmd get s3://duma-datasets/$FILE ~/2xar/ws/$FILE || exit 1
	fi
done

$verbose && echo "$banner Removing pyc files..."
cd $repos
find . -name '*.pyc' -delete

cd $repos/web1

$verbose && echo "$banner Building dev directories..."

# let path_helper build links and directories
./path_helper.py --dev-links || exit 1

$verbose && echo "$banner Setting up database..."

if $djangodb
then
    # We setup mysql with a default password, which is stored on S3.
    # Our mysql is never configured for remote access, so the password doesn't
    # _really_ matter, but it really prefers to have one, so set it up.
    if [[ ! -f ~/.my.cnf ]]; then
        # We are either setting up a new machine, or upgrading someone who
        # had no password set before.
        echo "Setting up mysql with a password"
        s3cmd get s3://2xar-duma-keys/my.cnf /tmp/.my.cnf
        PW="$(grep 'password' /tmp/.my.cnf | cut -f2 -d'=')"
		# Make sure it's actually running; in a container it won't startup automatically.
		sudo service mysql start
		sudo mysql <<EOL
use mysql
ALTER USER 'root'@'localhost' IDENTIFIED WITH mysql_native_password by '${PW}';
\q
EOL
		sudo service mysql restart
        mv /tmp/.my.cnf ~/.my.cnf

    fi
    if [[ -d '/home/www-data' ]]; then
        sudo cp ~/.my.cnf /home/www-data/.my.cnf
    fi

    # Verify that auth works.
	echo '\q' | mysql -u root
	if [ $? != 0 ]; then
        echo "Couldn't access mysql, check the setup"
        exit 1
    fi
	# create db if necessary
	echo '\q' | mysql -u root $dbname
	if [ $? != 0 ]
	then
		newdb=true
	fi

	if $newdb
	then
		echo "drop database if exists web1" | mysql -u root
		echo "create database web1 character set utf8mb4 collate utf8mb4_unicode_520_ci" | mysql -u root
	fi

	# run syncdb
	./manage.py migrate
	if [ $? != 0 ]; then
		echo "syncdb failed; check Django setup"
		exit 1
	fi
	# make sure site table is set up
	# This is needed to generate the QR code for 2FA.
	# Since non-production databases will typically be overridden by
	# production snapshots anyway (and will authenticate with production
	# tokens), don't worry about making this match the certificate
	# hostname in non-production apache setups.
	mysql -u root $dbname << END_MARKER
update django_site set domain='platform.twoxar.com' where id=1;
update django_site set name='platform.twoxar.com' where id=1;
END_MARKER
	# don't let sessions table get too big
	./manage.py clearsessions

	# Make sure we have the latest resource totals.
	python $repos/install/update_resources.py
fi

$verbose && echo "$banner Checking RNA-seq processing pipeline..."
if $rnaseq
then
	echo "$banner Installing RNA-seq pipeline..."
	cd $repos/web1/R/RNAseq
	make install
	make clean
	cd ../../
else
	echo "$banner RNA-seq pipeline does not need to be installed"
fi

$verbose && echo "$banner Setting up crontab..."

# Crontab configuration
# Required parameters are:
# - 'script' is the bare name of a script in the install directory; it also
#   is used to generate the corresponding log file name
# - 'when' is the cron time configuration string
# Optional parameters are:
# - 'parms' is anything to be added as parameters to the script
# - 'shell' is anything that goes before the script name in the crontab entry;
#   this can be a shell, or a wrapper program, or combinations of the two
#   - venv_py can be used as a wrapper to run a python program in the conda env
#   - as_www_data is a wrapper to run the program as the www-data user
as_www_data=$repos/install/as_www_data.sh
[ -e crontab.tmp ] && rm crontab.tmp
cronjob() {
	script="$1"
	when="$2"
	parms="$3"
	shell="$4"
	base=`echo $script | sed -e 's/\..*//'`
	rotate_log $base.log /var/log
	log=/var/log/$base.log
	sudo touch $log
	sudo chown $username $log
	script_dir=$repos/install
	echo "$when $script_dir/monitor_job.py $shell $script_dir/$script $parms >> $log 2>&1" >> crontab.tmp
}
# NOTE these operate in pacific time.
$shutdown && cronjob check_idle.py "* * * * *"
$checkdisk && cronjob check_disk.sh "*/10 * * * *" "${checkdisk_user}"
$backup && cronjob backup.sh "30 8,20 * * *"
$apache && cronjob apache.sh "0 2 * * 1,5"
$machinestats && cronjob check_machine_stats.py "*/5 * * * *" "" "$venv_py"
$update_resources && cronjob update_resources.py "@reboot" "" "$venv_py"
$check_aws && cronjob check_aws.py "50 * * * *" "" "$venv_py"
$check_aws && cronjob check_inst.py "0 8 * * *" "" "$venv_py"
$check_aws && cronjob check_apikeys.py "0 5 * * *" "" "$venv_py"
# run check_lts at 5am; this is currently 2 hours after the post test,
# when the system is most likely to be idle; the scan takes ~2 hours
$check_lts && cronjob check_platform_lts.py "0 5 * * *" "" "$as_www_data $venv_py"

if [ -s crontab.tmp ]
then
	crontab < crontab.tmp
	rm crontab.tmp
fi

# check for PROPS
false && if [ ! -d /usr/local/lib/R/site-library/PROPS ]
then
    echo "IN ORDER TO INSTALL PROPS:
    # get package from s3 and unzip:
    sudo s3cmd get s3://duma-datasets/PROPS_0.99.6.tar.gz || exit 1
    sudo tar -xzvf PROPS_0.99.6.tar.gz
    # then in an R session in the same folder:
    install.packages('bnlearn')
    install.packages('./PROPS', repos = NULL, type = 'source')"
	exit 1
fi


echo "Clearing cached S3 directory listings"
$repos/web1/path_helper.py --clear-s3-caches || exit 1

if $apache
then
	$verbose && echo "$banner Setting up apache..."
	if ! $repos/install/sub_install_apache.sh; then
	    echo "Failed to setup apache"
	    exit 1
	fi
	id_rsa=/var/www/.ssh/id_rsa
	fix='authkeys.py --apache'
	type="INSTALL (w/apache)"
else
	id_rsa=~/.ssh/id_rsa
	fix='authkeys.py'
	type="INSTALL"
fi
$verbose && echo "$banner $mode $type COMPLETED SUCCESSFULLY"
if sudo test \! -e $id_rsa
then
	echo "WARNING: no $id_rsa file; install with $fix"
fi

