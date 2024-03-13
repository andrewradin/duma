from django.utils.deprecation import MiddlewareMixin
from django.contrib.auth.models import User

"""
There are a mixture of old-style and new-style middlewares here.
Old-style middlewares (which are no longer directly supported in 2.2+) use
a provided MiddlewareMixin to become compatible with the new system.
"""
import logging
logger = logging.getLogger(__name__)


class FdUsageMiddleware(MiddlewareMixin):
    def process_request(self, request):
        pass
    def process_response(self, request, response):
        import os
        pid = os.getpid()
        import psutil
        nfds = psutil.Process(pid).num_fds()
        import logging
        logger = logging.getLogger(self.__class__.__name__)
        logger.warning('pid %d page %s %d fds',
                        pid,
                        request.META['PATH_INFO'],
                        nfds,
                        )
        return response

class MemoryUsageMiddleware(MiddlewareMixin):
    attr_key = '_mem'
    K = 1024
    threshold = 2*K*K
    def fmt_size(self,size):
        return "%dM" % (size/(self.K*self.K))
    def old_fmt_size(self,size):
        units='BKMGTP'
        idx=0
        while size > self.K:
            size /= self.K
            idx += 1
        return "%d%s" % (size,units[idx])
    def get_mem(self):
        import os
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss
    def process_request(self, request):
        setattr(request,self.attr_key,self.get_mem())
    def process_response(self, request, response):
        if hasattr(request,self.attr_key):
            old_mem = getattr(request,self.attr_key)
            new_mem = self.get_mem()
            diff = new_mem - old_mem
            if diff > self.threshold:
                import logging
                logger = logging.getLogger(self.__class__.__name__)
                import os
                logger.warning('pid %d page %s memory usage %s (total %s)',
                            os.getpid(),
                            request.META['PATH_INFO'],
                            self.fmt_size(diff),
                            self.fmt_size(new_mem),
                            )
        return response

import datetime
class ServiceTimeMiddleware(MiddlewareMixin):
    attr_key = '_timestamp'
    threshold = datetime.timedelta(0,3)
    now = datetime.datetime.now
    def process_request(self, request):
        setattr(request,self.attr_key,self.now())
    def process_response(self, request, response):
        if hasattr(request,self.attr_key):
            starttime = getattr(request,self.attr_key)
            elapsed = self.now() - starttime
            import logging
            logger = logging.getLogger(self.__class__.__name__)
            import os
            if elapsed > self.threshold:
                level = logging.WARN
            else:
                level = logging.DEBUG
            logger.log(level, 'from %s page %s elapsed %s',
                        request.META['REMOTE_ADDR'],
                        request.META['PATH_INFO'],
                        elapsed,
                        )
        return response


class IPMiddleware(MiddlewareMixin):
    @classmethod
    def ip_lookup_geo(cls, ip):
        ip = str(ip)
        if ip == '127.0.0.1' or ip == 'localhost':
            # Don't bother the API with localhost lookups.
            return None

        import os
        if 'PYTEST_CURRENT_TEST' in os.environ:
            # Don't waste our API usage on tests.
            return None
        import requests
        # This is a free access key, which is limited to ~10K lookups / month.
        # Should be way more than we need.
        # If we do ever update this to a paid one for some reason, move out of source control.
        # (Also, the paid version includes the ASN data we get separately now).
        url = f'http://api.ipstack.com/{ip}?access_key=7d0d0aafe58ca0b19ef4b00f31467724'
        try:
            return requests.get(url, timeout=2).json()
        except Exception as e:
            logger.info(f"Failed to geo lookup IP {ip}: {e}")
            return None

    @classmethod
    def ip_lookup_asn(cls, ip):
        ip = str(ip)
        if ip == '127.0.0.1' or ip == 'localhost':
            # Don't bother the API with localhost lookups.
            return None

        import os
        if 'PYTEST_CURRENT_TEST' in os.environ:
            # Don't waste our API usage on tests.
            return None

        # This is a totally free service that gives only ASN data.
        import requests
        url = f'https://api.iptoasn.com/v1/as/ip/{ip}'
        try:
            return requests.get(url, timeout=2).json()
        except Exception as e:
            logger.info(f"Failed to asn lookup IP {ip}: {e}")
            return None
    
    @classmethod
    def fetch_ip_details(cls, ip):
        geo_data = cls.ip_lookup_geo(ip) or {}
        asn_data = cls.ip_lookup_asn(ip) or {}
        region = geo_data.get("region_code")
        country = geo_data.get("country_code")
        city = geo_data.get("city")
        desc = asn_data.get("as_description")

        return f'{city}, {region}, {country} [{desc}]'

    def process_request(self,request):
        path = request.META['PATH_INFO']
        verified = request.user.is_verified()
        if path.startswith('/account/') or verified:
            access = 'normal'
        else:
            access = 'unverified'
        host = request.META['REMOTE_ADDR']
        user = request.user
        if not user.is_active:
            #assert path == '/account/login/'
            # actually, the path can be anything, but it will get redirected
            # if the user isn't logged in
            # XXX if we allowed a null user id, we could also report
            # XXX login attempts from new IP addresses
            return
        from browse.models import UserAccess
        ua,new = UserAccess.objects.get_or_create(
                user = user,
                host = host,
                access = access,
                )
        # UserAccess objects can be deleted from the users page, which
        # will trigger a monitoring alert on the next access of that type
        #print request.user.username,host,access,path,verified,new
        if new:
            import os
            if 'PYTEST_CURRENT_TEST' in os.environ:
                # We're running as part of pytest, we shouldn't be hitting slack.
                # Tests should all add appropriate UserAccess entries before login.
                raise Exception(f"""
                    pytest missing UserAccess entry for: "{user}" "{access}" "{host}"
                    All current entries: {UserAccess.objects.all().values()}
                    """)
            
            ip_details = self.fetch_ip_details(host)
            # post any unusual access to slack
            msg = 'on %s user %s %s access from new location %s (%s)' % (
                    os.uname()[1],
                    user.username,
                    access,
                    host,
                    ip_details
                    )
            from dtk.alert import slack_send
            slack_send(msg)


CONSULTANT_URL_RE_WHITELIST = [
    # Note that login doesn't need to be whitelisted because it runs in
    # the auth middleware which precedes this one.
    r'/consultant/.*$',
    r'/logout/$',
    r'/account/two_factor/.*$',
    ]
def restricted_access_middleware(get_response):
    def middleware(request):
        from django.http import HttpResponseRedirect, HttpResponse
        is_consultant = request.user.groups.filter(name='consultant').exists()
        if is_consultant:
            allowed = False
            import re
            for url_re in CONSULTANT_URL_RE_WHITELIST:
                # Note that re.match forces a match at the start of the string.
                if re.match(url_re, request.path):
                    allowed = True
                    break

            if not allowed:
                if request.path != '/':
                    logger.warning(
                            "Consultant requested non-whitelisted URL %s",
                            request.path)
                return HttpResponseRedirect('/consultant/')
        
        if request.user.username == 'qa':
            # The qa user can only access from localhost or from selenium machine, as it is
            # not configured for 2FA.
            addr = request.META['REMOTE_ADDR']
            if addr not in ['18.144.178.254', '127.0.0.1']:
                return HttpResponse('Forbidden from ' + addr, status=403)

        # All is fine, pass through to next middleware layer.
        response = get_response(request)

        if is_consultant and response.status_code >= 400:
            # Override our super-detailed error pages for consultants.
            import datetime
            msg = f"Something went wrong ({response.status_code}, {datetime.datetime.now()})"

            return HttpResponse(msg, status=response.status_code)

        return response

    return middleware

def errorpage_middleware(get_response):
    """Prevents any error pages from showing up if you're not logged in."""
    def middleware(request):
        from django.http import HttpResponse
        response = get_response(request)
        if response.status_code >= 400:
            if not hasattr(request, 'user') or not request.user.is_authenticated:
                return HttpResponse("Not Authenticated", status=403)
        return response
    return middleware



def msoffice_middleware(get_response):
    """Works around an office bug/feature that forces login when clicking platform links.
    
    See https://stackoverflow.com/questions/2653626/why-are-cookies-unrecognized-when-a-link-is-clicked-from-an-external-source-i-e

    Office attempts to prefetch the URL clicked, but doesn't have the appropriate cookie, so gets redirected to login - it then opens
    that login URL in the user's web browser.

    Instead, detect an Office-originated request via useragent, and if detected, return a stub HTML page that just refreshes itself, rather
    than the default of redirecting to the login page.  This will force Office to open the correct URL here.
    """
    import re
    office_re = re.compile(r'(Word|Excel|PowerPoint|ms-office|msoffice)')
    from django.http import HttpResponse
    meta_refresh = "<html><head><meta http-equiv='refresh' content='0'/></head><body></body></html>"
    def middleware(request):
        ua = request.META.get('HTTP_USER_AGENT', None)
        # logger.info("MSOffice middleware sees %s with ua %s", request.path, ua)

        if ua and office_re.search(ua):
            logger.info(f"MSOffice middleware triggered on {request.path} with ua {ua}, activating")
            return HttpResponse(meta_refresh, status=200)

        response = get_response(request)
        return response
    return middleware


from django.dispatch import receiver
from axes.signals import user_locked_out
@receiver(user_locked_out)
def on_user_lockout(sender, request, username, ip_address, **kwargs):
    import os
    from dtk.alert import slack_send
    auth_username = request.POST.get('auth-username', None)
    host = os.uname()[1]
    slack_send(f"Too many failed login requests to {host}, locking out: {username} {ip_address} {auth_username}")


from django_otp.plugins.otp_totp.models import TOTPDevice
from two_factor.models import PhoneDevice
from django.db.models.signals import pre_delete, post_save, pre_save
@receiver(post_save, sender=TOTPDevice)
@receiver(post_save, sender=PhoneDevice)
def on_2fa_change(sender, instance, created, **kwargs):
    # TOTP model updates on every token login, so only log creation.
    if created:
        from dtk.alert import slack_send
        import os
        host = os.uname()[1]
        slack_send(f'2FA device added {instance.name} for {instance.user.username} on {host}')

@receiver(pre_delete, sender=TOTPDevice)
@receiver(pre_delete, sender=PhoneDevice)
def on_2fa_remove(sender, instance, **kwargs):
    import os
    host = os.uname()[1]
    from dtk.alert import slack_send
    slack_send(f'2FA device removed {instance.name} for {instance.user.username} on {host}')


# This needs to be pre-save so we can pull the old password (hash)
@receiver(pre_save, sender=User)
def record_password_change(sender, **kwargs):
    user = kwargs.get('instance', None)
    msg = None
    if user:
        new_password = user.password
        try:
            old_password = User.objects.get(pk=user.pk).password
            if new_password != old_password:
                msg = f"Password changed for '{user.username}'"
        except User.DoesNotExist:
            msg = f"New user '{user.username}' created"
        

        if msg:
            import os
            from dtk.alert import slack_send
            host = os.uname()[1]
            slack_send(f'{msg} on {host}')
