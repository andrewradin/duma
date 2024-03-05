# Needs venv; crontab entry should run under correct py3

# NOTE:
# cloudwatch metrics are organized first into namespaces, and then into metric
# names below that. The custom disk space metrics below have their own
# namespace ('2xar/MachineStats') rather than existing in the standard EC2
# or EBS namespaces. If you know this, they're easy to find under cloudwatch
# on the AWS console.
#
# You can list all available metrics programatically as follows:
#
# from dtk.aws_api import AwsBoto3Base
# cw = AwsBoto3Base().session.client('cloudwatch')
# result = set()
# for response in paginator.paginate():
#     for d in response['Metrics']:
#         result.add((d['Namespace'],d['MetricName']))
# print(result)
#
# See experiments/dashboard/get_machine_stats.py for an example of accessing
# these metrics.

import boto3

def get_disk_space():
    import psutil
    import shutil

    out = {}
    for partition in psutil.disk_partitions():
        mount = partition.mountpoint
        if '/snap/' in mount:
            # Weird new ubuntu stuff, ignore.
            continue
        total, used, free = shutil.disk_usage(mount)
        out[mount] = dict(total=total, used=used, free=free)
    return out 

def push_disk_space(data, cw):
    import platform
    hostname = platform.node()
    metrics = []
    for name, space in data.items():
        for space_type, value in space.items():
            metrics.append({
                'MetricName': 'DiskSpace',
                'Dimensions': [{
                        'Name': 'Mount',
                        'Value': name,
                    }, {
                        'Name': 'SpaceType',
                        'Value': space_type,
                    }, {
                        'Name': 'Hostname',
                        'Value': hostname,
                    }],
                'Unit': 'Bytes',
                'Value': value
                })
    cw.put_metric_data(
            MetricData=metrics,
            Namespace='2xar/MachineStats',
            )
    

def main():
    from dtk.aws_api import AwsBoto3Base
    cw = AwsBoto3Base().session.client('cloudwatch')
    disk_space = get_disk_space()
    push_disk_space(disk_space, cw)

if __name__ == "__main__":
    main()
