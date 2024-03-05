#!/usr/bin/env python3

"""
Currently the AWS CLI doesn't provide a method for signing PUT urls, so
we have to do it through the API.
"""

import os
import sys
import json
from argparse import ArgumentParser
from urllib.parse import urlencode, quote_plus

import boto3

def main():
    parser = ArgumentParser(description='')
    parser.add_argument('-b', '--bucket',
                        required=True,
                        help='Bucket to upload to')
    parser.add_argument('-p', '--path',
                        required=True,
                        help='Path to sign within the bucket')
    parser.add_argument('-e', '--expiry',
                        type=int,
                        default=3600,
                        help='Expiry (in seconds)')
    args = parser.parse_args()

    s3 = boto3.client('s3')

    if True:
        response = s3.generate_presigned_url(
            ClientMethod='put_object',
            Params={'Bucket': args.bucket, 'Key': args.path},
            ExpiresIn=args.expiry,
        )
        # Prints the URL to use for uploading.
        upload_url = response
        print(upload_url)

        # Probably use something like this for uploading:
        print(f"Upload command:\ncurl -o upload_log.txt --upload-file ./sm.zip '{upload_url}'")
    else:
        # TODO: This can be used to generate a presigned POST request, which gives a bit more flexibility,
        # but doesn't actually get around the 5G limitations, so isn't super useful right now.
        response = s3.generate_presigned_post(
            Bucket=args.bucket,
            Key=args.path,
            ExpiresIn=args.expiry,
        )

        out = 'curl'
        for k, v in response['fields'].items():
            out += f' -F "{k}={v}"'

        out += ' ' + response['url']


        # Probably use something like this for uploading:
        print(f"Upload command:\n{out}")



if __name__ == "__main__":
    main()
