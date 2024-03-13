

import runpy
from mock import patch

# NOTE: Nothing here is mocked out, so we are only testing things that don't
# write anything.

def test_status():
    with patch('sys.argv', ['', 'status']):
        runpy.run_module('aws_op', run_name='__main__')

def test_list_images():
    with patch('sys.argv', ['', 'list_images','--ubuntu-release','14.04']):
        runpy.run_module('aws_op', run_name='__main__')

def test_list_release():
    with patch('sys.argv', ['', 'list_releases']):
        runpy.run_module('aws_op', run_name='__main__')

def check_machine(mch,specs,disks,release):
    kwargs = mch.build_spec.get_create_parameters(mch)
    print("We got ", kwargs, " for ", mch, specs, disks)
    assert mch.build_spec.ubuntu_release == release
    from dtk.data import dict_subset
    assert specs == dict_subset(kwargs,specs.keys())
    output_disks = {x['DeviceName']: x['Ebs']['VolumeSize'] for x in kwargs['BlockDeviceMappings']}
    assert disks == output_disks

def check_named_machine(name,specs,disks,release):
    from aws_op import Machine
    mch = Machine.name_index[name]
    check_machine(mch,specs,disks,release)

def test_create_specs():
    check_named_machine('selenium',
            dict(
                    KeyName='twoxar-create-uswest1',
                    SubnetId='subnet-649c5301',
                    ),
            {'/dev/sda1':30},
            '20.04',
            )
    check_named_machine('dev08',
            dict(
                    KeyName='twoxar-create-uswest2',
                    SubnetId='subnet-a314c0c6',
                    ),
            {'/dev/sda1':50,'/dev/sdf':500},
            '16.04',
            )
