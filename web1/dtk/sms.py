# See http://boto3.readthedocs.io/en/latest/reference/services/sns.html
# Check cloudwatch Logs for message delivery statuses
# messages arrive from short code 44154

class AwsSMS(object):
    @staticmethod
    def send_sms(device, token):
        from dtk.aws_api import AwsBoto3Base
        ctrl = AwsBoto3Base()
        session = ctrl.session
        client=session.client('sns',region_name=ctrl.region_name)
        rsp=client.publish(
                PhoneNumber=device.number.as_e164,
                Message=token,
                )

# To test:
# >>> from two_factor.models import PhoneDevice
# >>> d=PhoneDevice.objects.get(pk=something)
# >>> from dtk.sms import AwsSMS
# >>> AwsSMS.send_sms(d,'my token')

