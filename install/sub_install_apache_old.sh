#!/bin/bash

# root of git repository
repos=~/2xar/twoxar-demo

sudo apt-get install apache2 apache2-utils libapache2-mod-wsgi certbot python3-certbot-apache apache2-dev
sudo a2enmod wsgi
sudo a2enmod ssl

# mod_wsgi recommends installing via pip in your virtual env, so that you
# get a mod_wsgi that matches the python version of your virtual env.
python3 -m pip install mod_wsgi


if sudo [ ! -f "/etc/letsencrypt/live/platform.twoxar.com/fullchain.pem" ]; then
    # We need SSL certs before continuing with the rest of this process.
    # Otherwise, the apache.conf we put in place below will break apache
    # due to the certs and ssl configs it points to not existing.
    # And then once apache is broken, certbot can't (easily) generate certs
    # for you anymore, because it expects a working http server for its
    # challenge/response mechanism.
    echo "No SSL certs found, you'll want to set those up before continuing."
    echo "Run: "
    echo "   sudo certbot --apache"
    echo ""
    echo "And then rerun this install script after completing it."
    exit 1
fi


# configure apache files
sudo cp $repos/apache/apache2.conf /etc/apache2/apache2.conf

# wsgi.load points at the wsgi from our virtual env (rather than the system
# one which uses the default system version of python)
sudo cp $repos/apache/wsgi.load /etc/apache2/mods-available/wsgi.load

# envvars overrides LD_LIBRARY_PATH to point into the env
sudo cp $repos/apache/envvars /etc/apache2/envvars
sudo cp $repos/apache/django.wsgi /var/www/html/django.wsgi
sudo cp $repos/apache/default-ssl.conf /etc/apache2/sites-available
sudo a2ensite default-ssl

echo "Setting up /home/www-data"

# allow running specially-designated scripts
sudo cp $repos/install/sudoers /etc/sudoers.d/50-duma

www_root=/home/www-data/2xar
www_repos=$www_root/twoxar-demo

# copy things to web directory
sudo rm -rf $www_repos
sudo mkdir -p $www_repos
sudo chown www-data $www_root $www_repos
sudo cp -r $repos/moleculeSimilarity $www_repos
sudo cp -r $repos/web1 $www_repos
# lock down version in www-data copy of local settings file
# (since www-data doesn't have the full git repo)
vers=`$repos/web1/path_helper.py version`
echo "version='$vers'" > /tmp/version
sudo sh -c "cat /tmp/version >> $www_repos/web1/local_settings.py"
sudo chown -R www-data $www_repos

# prevent ssh error messages re: known hosts file updates
sudo mkdir /var/www/.ssh
sudo chmod 700 /var/www/.ssh
sudo chown www-data /var/www/.ssh

# lts requires gnupg
sudo mkdir /var/www/.gnupg
sudo chmod 700 /var/www/.gnupg
sudo chown www-data /var/www/.gnupg

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

