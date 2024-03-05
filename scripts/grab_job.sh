#!/bin/sh

# param is <ws_id>/<plugin>/<job_id>

JOBS=$@

for JOB in $JOBS
do
    echo "Grabbing job $JOB"
    cd ~/2xar/ws
    mkdir -p $JOB
    rmdir $JOB || exit 1

    # NOTE: the two references to 'platform' below assume that you have ssh
    # configured so that this routes to the ubuntu login on platform.twoxar.com.
    # This can be accomplished by adding the following to ~/.ssh/config
    # (minus the comment characters at the start of each line):
    #Host platform
    #HostKeyAlias platform
    #HostName platform.twoxar.com
    #User ubuntu
    # From a machine inside the AWS VPC, replace platform.twoxar.com with
    # the internal IP address of the platform instance (at the time of this
    # writing, 172.31.43.174).  Machines outside AWS need to be listed in
    # the security group that authorizes ssh access.

    scp -r platform:/home/www-data/2xar/ws/$JOB $JOB

    if expr match $JOB '[0-9]*/path/'
    then
        PATHFILE=`echo $JOB | sed -e 's,path/\([0-9]*\),paths.\1.tsv.gz,'`
        scp -r platform:/home/www-data/2xar/ws/$PATHFILE $PATHFILE
    fi
done
