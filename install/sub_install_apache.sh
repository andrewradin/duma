#!/bin/bash

# root of git repository
repos=~/2xar/twoxar-demo

https_hostname=`$repos/web1/path_helper.py https_hostname`
if [ "$https_hostname" == "" ]
then
	echo ERROR: https_hostname must be set in local_settings.py
	exit 1
fi

sudo apt-get install apache2 apache2-utils libapache2-mod-wsgi certbot python3-certbot-apache apache2-dev
sudo a2enmod wsgi
sudo a2enmod ssl

# the previous version of apache configuration overwrote several standard
# files; to smooth migration, begin by putting the standard files back in place
# XXX this (and 20.04_unmodified_apache_config) can be removed after the first
# XXX deployment
# XXX Also, to allow easy fallback in case of deployment problems, the
# XXX old-style config files were left in twoxar-demo/apache and the previous
# XXX version of this script in sub_install_apache_old.sh. After successful
# XXX deployment, the old version of this script and everying in ../apache
# XXX except wsgi.load and duma.conf can be removed.
src=$repos/install/20.04_unmodified_apache_config
sudo cp $src/apache2.conf /etc/apache2
sudo cp $src/envvars /etc/apache2
sudo cp $src/default-ssl.conf /etc/apache2/sites-available
sudo cp $src/wsgi.load /etc/apache2/mods-available
sudo rm /var/www/html/django.wsgi
sudo a2dissite default-ssl

# mod_wsgi recommends installing via pip in your virtual env, so that you
# get a mod_wsgi that matches the python version of your virtual env.
python3 -m pip install mod_wsgi


if sudo [ ! -f "/etc/letsencrypt/live/$https_hostname/fullchain.pem" ]; then
    # We need SSL certs before continuing with the rest of this process.
    # Otherwise, the apache.conf we put in place below will break apache
    # due to the certs and ssl configs it points to not existing.
    # And then once apache is broken, certbot can't (easily) generate certs
    # for you anymore, because it expects a working http server for its
    # challenge/response mechanism.
    echo "No SSL certs found, you'll want to set those up before continuing."
    echo "Make sure $https_hostname is in DNS and then run: "
    echo "   sudo certbot --apache"
    echo ""
    echo "And then rerun this install script after completing it."
    exit 1
fi


# wsgi.load points at the wsgi from our virtual env (rather than the system
# one which uses the default system version of python)
sudo cp $repos/apache/wsgi.load /etc/apache2/mods-available/wsgi.load

sudo cp $repos/apache/duma.conf /etc/apache2/sites-available
sudo a2ensite duma
sudo a2ensite 000-default-le-ssl

echo "Setting up /home/www-data"

# allow running specially-designated scripts
sudo cp $repos/install/sudoers /etc/sudoers.d/50-duma

www_root=/home/www-data/2xar
www_repos=$www_root/twoxar-demo

# make sure www_root and lts subdir exist
sudo mkdir -p $www_root/lts
sudo chown www-data $www_root $www_root/lts

# link source under www_root
sudo rm -rf $www_repos
sudo ln -s $repos $www_repos

# provide link to wsgi.py script
sudo ln -s $www_repos/web1/web1/wsgi.py /var/www/html

# prevent ssh error messages re: known hosts file updates
sudo mkdir /var/www/.ssh
sudo chmod 700 /var/www/.ssh
sudo chown www-data /var/www/.ssh

# lts requires gnupg
sudo mkdir /var/www/.gnupg
sudo chmod 700 /var/www/.gnupg
sudo chown www-data /var/www/.gnupg

# git now requires specific authorization to allow www-data to
# operate on a git repo it doesn't own. The concern is some bad actor
# will plant a .git directory somewhere and you'll inadvertantly
# run their trigger scripts. See:
# https://github.blog/2022-04-12-git-security-vulnerability-announced/
sudo tee /var/www/.gitconfig >/dev/null << EOF
[safe]
	directory = /mnt2/ubuntu/twoxar-demo
EOF
sudo chown www-data /var/www/.gitconfig

# install css files and the like
sudo chmod 777 /var/www/html
$repos/web1/manage.py collectstatic
sudo chmod 755 /var/www/html
sudo chown -R www-data /var/www/html

# make sure publish link is set to serve dynamic content
sudo mkdir -p /var/www/html/publish
sudo chown www-data /var/www/html/publish
sudo ln -sTf /var/www/html/publish /home/www-data/2xar/publish

# Make sure production has all the things that it would have
# gotten from path_helper.py --dev-links, and didn't get
# copied above.
sudo -u www-data mkdir -p /home/www-data/2xar/ws || exit 1
for FILE in ~/2xar/ws/*
do
	if [ -f $FILE ]
	then
		sudo -u www-data cp $FILE $www_root/ws
	fi
done

sudo -u www-data $repos/web1/path_helper.py --prod-links || exit 1

# remove possibly stale S3-backed files to force the latest versions to load
sudo -u www-data $repos/web1/path_helper.py --clear-s3-caches || exit 1

# restart apache
sudo service apache2 graceful

