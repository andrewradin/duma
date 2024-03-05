#!/usr/bin/python3

import subprocess
import os

# see argparse description at bottom of file for details on usage

def configure_new_instance(hostname):
    if hostname:
        configure_hostname(hostname)
    configure_drive('DumaData','/mnt2')
    configure_links('ubuntu','/mnt2')
    print('''
POSSIBLE REMAINING MANUAL STEPS:

If the machine accesses other machines (e.g. LTS, workers):
- use "authkeys.py new_key_pair <hostname>" to create and install ssh keys
- add the machine name to the ssh list of the appropriate FirewallConfigs
- use "authkeys.py put_ssh <other_host>" to update any modified ssh
  authorized_keys files

If any custom access configuration is required, create a FirewallConfig
instance in dtk/aws_api.py for the instance name.

Install the source using:
  cd ~/2xar
  git clone https://github.com/twoXAR/twoxar-demo.git
  cd twoxar-demo
  # set machine_type in web1/local_settings.py
  ./install/install.sh

Install a database snapshot using:
  ~/2xar/twoxar-demo/scripts/import_prod_db.sh

For public access:
- allocate and attach an EIP
- add IP address to Route 53
- consider adding the EIP to the known_ips file; this is only required if the
  machine must appear in firewall rules to access machine in other VPCs
''')

def check(cmd):
    return subprocess.run(cmd,check=True)

def sudo(cmd):
    return check(['sudo']+cmd)

def lines_from(cmd):
    cmpl = subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.PIPE, #capture_output=True,
            )
    return cmpl.stdout.decode().split('\n')

def drive_inventory():
    labels = None
    result = {}
    # get data and filter to 'nvme' devices
    for line in lines_from(['lsblk','-i','-o','NAME,FSTYPE,LABEL,MOUNTPOINT']):
        if not labels:
            labels = line.split()
            offsets = [line.index(x) for x in labels]+[len(line)]
            continue
        values = [
                line[offsets[i]:offsets[i+1]].strip()
                for i in range(len(labels))
                ]
        values[0] = values[0].lstrip('`-') # remove -i mode tree drawing chars
        if not values[0].startswith('nvme'):
            continue
        result[values[0]] = dict(zip(labels,values))
    # for all nvmeXnYpZ devices, remove the nvmeXnY device (which holds the
    # partition table, which we don't want to overwrite, but otherwise appears
    # as a blank drive, which is what we're trying to find)
    to_remove = set()
    import re
    for key in result.keys():
        m = re.match(r'(nvme\d+n\d+)p\d+',key)
        if m:
            to_remove.add(m.group(1))
    for key in to_remove:
        try:
            del(result[key])
        except KeyError:
            pass
    return list(result.values())

def configure_drive(label,mount_point):
    # https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/nvme-ebs-volumes.html
    # You can't count on the /dev entry following the order in the instance
    # block device mapping, so here we need to search for a candidate data
    # drive.
    l = drive_inventory()
    def dump(inv):
        if not inv:
            print('Empty Inventory List')
        else:
            inv = sorted(inv,key=lambda x:x['NAME'])
            for d in inv:
                print(','.join([
                        f'{x[0]:x[1]}'
                        for x in zip(d.keys(),d.values())
                        ]))
    if not l:
        print('no AWS-type drives found')
        return
    labels = set([d['LABEL'] for d in l])
    root_label = 'cloudimg-rootfs'
    if root_label not in labels:
        print('no root drive found; got:')
        dump(l)
        return
    l = [d for d in l if d['LABEL'] != 'cloudimg-rootfs']
    if len(l) == 0:
        print('no extra drive found')
        return
    if len(l) != 1:
        print('multiple candidate drives found; got:')
        dump(l)
        return
    stat = l[0]
    dev = '/dev/'+stat['NAME']
    print('found data drive candidate on',dev)
    if stat['FSTYPE']:
        print('skipping mkfs; already formatted as',stat['FSTYPE'])
    else:
        sudo(['mkfs.ext4',dev])
    if stat['LABEL']:
        print('skipping e2label; already labeled as',stat['LABEL'])
    else:
        sudo(['e2label',dev,label])
    if stat['MOUNTPOINT']:
        print('skipping mount setup; already mounted as',stat['MOUNTPOINT'])
    else:
        fstab='/etc/fstab'
        cmpl = subprocess.run(['grep','-q',mount_point,fstab])
        if cmpl.returncode:
            fstab_entry='LABEL='+label+'\t'+mount_point+'\text4\tdefaults\t0\t0'
            sudo(['sh','-c','echo "'+fstab_entry+'" >>'+fstab])
            # without f-strings, this runs on 16.04 /usr/bin/python3 (3.5.2)
            #fstab_entry = f'LABEL={label}\t{mount_point}\text4\tdefaults\t0\t0'
            #sudo(['sh','-c',f'echo "{fstab_entry}" >>{fstab}'])
        sudo(['mkdir','-p',mount_point])
        sudo(['mount',mount_point])

def configure_hostname(hostname):
    if lines_from(['hostname'])[0] == hostname:
        print('hostname already set to',hostname)
    else:
        sudo(['sh','-c','echo "'+hostname+'" >>/etc/hostname'])
        sudo(['hostname',hostname])
    hostsfile='/etc/hosts'
    cmpl = subprocess.run(['grep','-q',hostname,hostsfile])
    if cmpl.returncode == 0:
        print(hostname,'already in',hostsfile)
    else:
        sudo(['sed','-i','1a127.0.0.1 '+hostname,hostsfile])
    for name,value in [
            ('user.name','ubuntu@'+hostname),
            ('user.email','some_user@twoxar.com'),
            ]:
        try:
            old = lines_from(['git','config','--global',name])[0]
        except subprocess.CalledProcessError:
            old = ''
        if old:
            print('git',name,'already set to',old)
        else:
            check(['git','config','--global',name,value])
            print('setting git',name,'to',value,'(modify manually if desired)')

def configure_links(user,remote):
    src = os.path.join('/home',user,'2xar')
    check(['mkdir','-p',src])
    if not os.path.isdir(remote):
        print('skipping link configuration; no',remote)
        return
    dest = os.path.join(remote,user)
    sudo(['mkdir','-p',dest])
    sudo(['chown',user,dest])
    for subdir in ('lts','publish','ws','twoxar-demo'):
        spath = os.path.join(src,subdir)
        if os.path.exists(spath):
            print('skipping existing',spath)
            continue
        dpath = os.path.join(dest,subdir)
        check(['mkdir','-p',dpath])
        subprocess.run(['ln','-s',dpath,'.'],cwd=src,check=True)

if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser(
            formatter_class=argparse.RawDescriptionHelpFormatter,
            description='''Initialize a new instance

This establishes the basic 2-disk layout. It essentially does the same thing
as disk_layout.py, but on a bare machine without relying on the virtualenv
or any other files in the repo. It has some basic protections against
clobbering things if run on an already-installed machine.

To get to the point where you can run this, you need to:
- add the new machine to aws_op.py
- run aws_op.py -m <machine_name> create
- run authkeys.py --pem twoxar-create-<region>.pem put_ssh <machine_name>
- get public IP using aws_op.py -m <machine_name> show_ips
- verify ability to ssh to ubuntu@<public_ip>
- scp this script to the new machine (the home directory is fine)
- ssh to the new machine and run the script
''',
            )
    parser.add_argument('--hostname')
    args = parser.parse_args()

    configure_new_instance(args.hostname)
