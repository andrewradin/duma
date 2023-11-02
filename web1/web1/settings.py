# Django settings for web1 project.
from path_helper import PathHelper

DEBUG = not PathHelper.is_production()

ADMINS = (
    # ('Your Name', 'your_email@example.com'),
)
from django.urls import reverse_lazy
LOGIN_URL = 'two_factor:login'
LOGIN_REDIRECT_URL='two_factor:profile'
#LOGIN_REDIRECT_URL='/' # XXX revert to this once everyone's configured?
TWO_FACTOR_SMS_GATEWAY='dtk.sms.AwsSMS'

MANAGERS = ADMINS

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.mysql', # Add 'postgresql_psycopg2', 'mysql', 'sqlite3' or 'oracle'.
        'NAME': 'web1',                      # Or path to database file if using sqlite3.
        'HOST': '',                      # Set to empty string for localhost. Not used with sqlite3.
        'PORT': '',                      # Set to empty string for default. Not used with sqlite3.
        'OPTIONS': {
            'charset': 'utf8mb4',
            'read_default_file': PathHelper.home + '/.my.cnf',
            # In most cases you won't need nearly this many, but e.g. multirefresh on high # of cores it can.
            # The failure mode when we run out of connections can require manual recovery, so avoid.
            # The default is like 150 or something very small like that.
            "init_command": "SET GLOBAL max_connections = 10000",
            },
        'TEST': {
                'CHARSET': 'utf8mb4',
                # According to https://stackoverflow.com/questions/51278467/mysql-collation-utf8mb4-unicode-ci-vs-utf8mb4-default-collation
                # the best collation to use is this, until we get to mysql8.
                'COLLATION': 'utf8mb4_unicode_520_ci'
        }
    }
}

# django 3.2 requires this to insure backward compatability without warnings
DEFAULT_AUTO_FIELD = 'django.db.models.AutoField'

# -- Security-related additions -- #
# See https://docs.djangoproject.com/en/2.1/ref/middleware/#http-strict-transport-security
# or https://adamj.eu/tech/2019/04/10/how-to-score-a+-for-security-headers-on-your-django-website/
# for descriptions & details.

# Attempts to detect user-specified JS from being sent to the server & returned for rendering.
SECURE_BROWSER_XSS_FILTER = True

# This one will only take effect in production, because it only gets set on the first https
# request (after which your browser will reject all https requests)
# Be careful, this tells browsers to never use HTTP on this domain for X seconds, which will
# completely break a site that doesn't have working HTTPS.
# To be useful this has to be a very high number.  Currently set to 90 days (in seconds).
SECURE_HSTS_SECONDS = 90 * 24 * 60 * 60
# We don't want this one; we don't use subdomains beyond platform, and
# according to https://serverfault.com/questions/482350/can-i-turn-on-hsts-for-1-subdomain
# it might turn it on for non-platform subdomains.
SECURE_HSTS_INCLUDE_SUBDOMAINS = False

# Disables content-type sniffing, which can be used for XSS.
SECURE_CONTENT_TYPE_NOSNIFF = True

# Prevents leaking internal URLs to external sites.
# Currently rely on django-referrer-policy package for this, but it is baked into django 3+,
# so switch when we upgrade.
# We also set this via a meta tag in our base HTML, but doing it via headers is also good.
REFERRER_POLICY='same-origin'

# Disable embedding us as an iframe.
X_FRAME_OPTIONS = 'SAMEORIGIN'

# Explicit allow-list of sites that we can load content from.
# - unsafe-inline here isn't ideal, but it's going to be a bunch of work to do it properly,
#   either moving inline scripts into files, or adding nonces/hashes to them.
# - unsafe-eval is required for plotly - see https://github.com/plotly/plotly.js/issues/897
CSP_DEFAULT_SRC = (
    "'self'",
    "'unsafe-inline'",
    "'unsafe-eval'",
    'cdn.datatables.net',
    'plotly.com',
    'cdn.plot.ly',
    'stackpath.bootstrapcdn.com',
    'cdnjs.cloudflare.com/ajax/libs/cytoscape/',
    )
# I believe plotly is generating these data: URIs, but it is hard to tell.
# e.g. see retrospective page.
CSP_IMG_SRC = ("data:", *CSP_DEFAULT_SRC)

# Cookie-related options.
SESSION_COOKIE_HTTPONLY = True
CSRF_COOKIE_HTTPONLY = True
if PathHelper.is_apache:
    SESSION_COOKIE_SECURE = True
    CSRF_COOKIE_SECURE = True
    # The "__Secure-" prefix has special meaning to modern browsers and is more secure.
    SESSION_COOKIE_NAME = "__Secure-sessionid"
    CSRF_COOKIE_NAME = "__Secure-csrftoken"

# This will expire sessions after 7 days.
SESSION_COOKIE_AGE = 7 * 24 * 60 * 60

# -- End Security-related additions -- #


# django-axes login ratelimiting config

# Setting this quite leniently, 20 attempts / 5 minutes.
# Mostly just trying to prevent brute-force, which needs a lot more.  We also
# alert if this is happening.
from datetime import timedelta
AXES_USERNAME_FORM_FIELD = 'auth-username'
AXES_PASSWORD_FORM_FIELD = 'auth-password'
AXES_FAILURE_LIMIT = 20
AXES_COOLOFF_TIME = timedelta(minutes=15)  # Reset failures after this time.
AXES_RESET_ON_ACCESS = True # Reset failure attempts on successful login



# This got stricter with 1.8
ALLOWED_HOSTS = ['localhost','127.0.0.1','::1']
if PathHelper.cfg('https_hostname'):
    ALLOWED_HOSTS += [PathHelper.cfg('https_hostname')]

# Local time zone for this installation. Choices can be found here:
# http://en.wikipedia.org/wiki/List_of_tz_zones_by_name
# although not all choices may be available on all operating systems.
# In a Windows environment this must be set to your system time zone.
TIME_ZONE = 'America/Los_Angeles'

# Language code for this installation. All choices can be found here:
# http://www.i18nguy.com/unicode/language-identifiers.html
LANGUAGE_CODE = 'en-us'

SITE_ID = 1

# If you set this to False, Django will make some optimizations so as not
# to load the internationalization machinery.
USE_I18N = True

# If you set this to False, Django will not format dates, numbers and
# calendars according to the current locale.
USE_L10N = True

# If you set this to False, Django will not use timezone-aware datetimes.
USE_TZ = True

# Absolute filesystem path to the directory that will hold user-uploaded files.
# Example: "/home/media/media.lawrence.com/media/"
#MEDIA_ROOT = PathHelper.publish

# URL that handles the media served from MEDIA_ROOT. Make sure to use a
# trailing slash.
# Examples: "http://media.lawrence.com/media/", "http://example.com/media/"
#MEDIA_URL = '/publish/'

# TODO: We should look into the caching behavior of these static files and make
# sure that it is appropriate.
# In particular, they should probably be served with a no-cache header so that
# we don't use stale copies on the rare occasion we update them.

# Absolute path to the directory static files should be collected to.
# Don't put anything in this directory yourself; store your static files
# in apps' "static/" subdirectories and in STATICFILES_DIRS.
# Example: "/home/media/media.lawrence.com/static/"
STATIC_ROOT = '/var/www/html/static/'

# URL prefix for static files.
# Example: "http://media.lawrence.com/static/"
STATIC_URL = '/static/'

# Additional locations of static files
STATICFILES_DIRS = (
    # Put strings here, like "/home/html/static" or "C:/www/django/static".
    # Always use forward slashes, even on Windows.
    # Don't forget to use absolute paths, not relative paths.
)

# List of finder classes that know how to find static files in
# various locations.
STATICFILES_FINDERS = (
    'django.contrib.staticfiles.finders.FileSystemFinder',
    'django.contrib.staticfiles.finders.AppDirectoriesFinder',
#    'django.contrib.staticfiles.finders.DefaultStorageFinder',
)

# Make this unique, and don't share it with anybody.
SECRET_KEY = '(#(gr#0_4vy#2%nt%s^q#dwv*c*_$wxvuw)@e6%&amp;2y!oeg!@ue'

# use settings.py when looking for form/widget templates
# (thanks to https://www.caktusgroup.com/blog/2018/06/18/make-all-your-django-forms-better/)
FORM_RENDERER = 'django.forms.renderers.TemplatesSetting'

TEMPLATES=[
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [
            PathHelper.website_root + 'templates',
        ],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                "django.contrib.auth.context_processors.auth",
                "django.template.context_processors.debug",
                "django.template.context_processors.i18n",
                "django.template.context_processors.media",
                "django.template.context_processors.request",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]

MIDDLEWARE = (
    'django.middleware.security.SecurityMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
    'django_referrer_policy.middleware.ReferrerPolicyMiddleware',
    'csp.middleware.CSPMiddleware',
    #'dtk.middleware.FdUsageMiddleware', # fd monitor
    'dtk.middleware.MemoryUsageMiddleware', # memory monitor
    'dtk.middleware.ServiceTimeMiddleware', # slow page monitor
    'django.middleware.common.CommonMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.http.ConditionalGetMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'dtk.middleware.errorpage_middleware', # Don't show error pages if not logged in
    'dtk.middleware.restricted_access_middleware',
    'dtk.middleware.msoffice_middleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    # Uncomment the next line for simple clickjacking protection:
    # 'django.middleware.clickjacking.XFrameOptionsMiddleware',
    'django_otp.middleware.OTPMiddleware',
    'dtk.middleware.IPMiddleware', # access monitor
    'pyinstrument.middleware.ProfilerMiddleware',
    'axes.middleware.AxesMiddleware', # login ratelimit.
)

ROOT_URLCONF = 'web1.urls'

# Python dotted path to the WSGI application used by Django's runserver.
WSGI_APPLICATION = 'web1.wsgi.application'

INSTALLED_APPS = (
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.sites',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    # Uncomment the next line to enable the admin:
    'django.contrib.admin',
    # Uncomment the next line to enable admin documentation:
    # 'django.contrib.admindocs',
    'django_otp',
    'django_otp.plugins.otp_static',
    'django_otp.plugins.otp_totp',
    'two_factor',
    # Django login ratelimiting
    'axes',

    # Our apps.
    'browse',
    'runner',
    'notes',
    'drugs',
    'nav',
    'flagging',
    'ctsearch',
    'ktsearch',
    'patsearch',
    'moldata',
    'ge',
    'rvw',
    'wsadmin',
    'wsmgr',
    'score',
    'xws',
    # must be last; this lets django find standard form/widget templates
    # after altering FORM_RENDERER above
    'django.forms',
)

# Django warns about cache keys that are incompatible with memcached; we don't use that,
# so we don't care.
import warnings
from django.core.cache import CacheKeyWarning
warnings.simplefilter("ignore", CacheKeyWarning)

CACHES = {
    'default': {
        'BACKEND': 'django.core.cache.backends.filebased.FileBasedCache',
        'LOCATION': PathHelper.django_cache,
        'TIMEOUT': 7*24*3600, # keep entries for a week
        'OPTIONS': {
            'MAX_ENTRIES': 200,
            'CULL_FREQUENCY': 3, # retain 67% of entries when culling
        }
    },
    'browse.utils.Scorebox': {
        'BACKEND': 'django.core.cache.backends.filebased.FileBasedCache',
        'LOCATION': PathHelper.django_cache+'scorebox',
        'TIMEOUT': 30*24*3600, # keep entries for a month
        'OPTIONS': {
            'MAX_ENTRIES': 20000, # 400 drugs x 50 scores
            'CULL_FREQUENCY': 5, # retain 80% of entries when culling
        }
    },
    'drugnotes': {
        'BACKEND': 'django.core.cache.backends.filebased.FileBasedCache',
        'LOCATION': PathHelper.django_cache+'drugnotes',
        'TIMEOUT': None, # Never expire; can revisit if needed
        'OPTIONS': {
            'MAX_ENTRIES': 1000000, # These are small, can keep lots.
            'CULL_FREQUENCY': 5, # retain 80% of entries when culling
        }
    },
    'selectability': {
        'BACKEND': 'diskcache.DjangoCache',
        'LOCATION': PathHelper.django_cache+'selectability',
        'TIMEOUT': 30*24*3600, # keep entries for a month
        'OPTIONS': {
        }
    },
    'enrichment_metric': {
        'BACKEND': 'diskcache.DjangoCache',
        'LOCATION': PathHelper.django_cache+'enrichment_metric',
        'TIMEOUT': 30*24*3600, # keep entries for a month
        'OPTIONS': {
        }
    },
    'cached_dict_elements': {
        'BACKEND': 'diskcache.DjangoCache',
        'LOCATION': PathHelper.django_cache+'cached_dict_elements',
        'TIMEOUT': 30*24*3600, # keep entries for a month
        'OPTIONS': {
            'size_limit': 2 ** 30 # 1 gigabyte
        }
    },
}

TEST_RUNNER = 'django.test.runner.DiscoverRunner'

# A sample logging configuration. The only tangible logging
# performed by this configuration is to send an email to
# the site admins on every HTTP 500 error when DEBUG=False.
# See http://docs.djangoproject.com/en/dev/topics/logging for
# more details on how to customize your logging configuration.
LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'filters': {
    },
    'formatters': {
        'normal': {
            'format': '%(asctime)s %(levelname)s %(name)s %(message)s',
        },
        # Console logging should be short, but informative.
        # Mostly just removing the full date - if you need that, you probably
        # want to be logging to a file anyway.
        'console': {
            'format': '%(asctime)s.%(msecs)03d %(name)s [%(levelname)s]: %(message)s',
            'datefmt': '%H:%M:%S',
        }
    },
    'handlers': {
        'syslog': {
            'level': 'DEBUG',
            'class': 'logging.handlers.SysLogHandler',
            'address': '/dev/log',
            'facility': 16, # local0
            'formatter': 'normal',
        },
        'console': {
            'level': 'INFO',
            'class': 'logging.StreamHandler',
            'formatter': 'console',
            'stream': 'ext://sys.stdout'
        },
    },
    'loggers': {
        '': {
            'handlers': ['syslog', 'console'],
            'level': 'DEBUG',
        },
        # set this level to DEBUG to see full SQL,
        # or something higher to suppress it
        'django.db.backends': {
            'handlers': ['syslog'],
            #'level': 'DEBUG',
            'level': 'ERROR',
            'propagate': False
        },
        # suppress urllib3 debug output
        'urllib3': {
            'handlers': ['syslog'],
            'level': 'INFO',
            'propagate': False
        },
        # suppress s3transfer debug output
        's3transfer': {
            'handlers': ['syslog'],
            'level': 'INFO',
            'propagate': False
        },
        # suppress boto debug output
        'botocore': {
            'handlers': ['syslog'],
            'level': 'INFO',
            'propagate': False
        },
        'boto3': {
            'handlers': ['syslog'],
            'level': 'INFO',
            'propagate': False
        },
        # suppress peewee debug output
        'peewee': {
            'handlers': ['syslog'],
            'level': 'INFO',
            'propagate': False
        },
        # suppress cache hit reporting
        'dtk.cache': {
            'handlers': ['syslog'],
            'level': 'INFO',
            'propagate': False
        },
        # suppress neo4j debug output
        'neobolt': {
            'handlers': ['syslog'],
            'level': 'INFO',
            'propagate': False
        },
        'neo4j': {
            'handlers': ['syslog'],
            'level': 'INFO',
            'propagate': False
        },
        'axes.watch_login': {
            'handlers': ['syslog'],
            'level': 'WARN',
            'propagate': False
        },
        'axes.apps': {
            'handlers': ['syslog'],
            'level': 'WARN',
            'propagate': False
        },
        'matplotlib.font_manager': {
            'handlers': ['syslog'],
            'level': 'WARN',
            'propagate': False
        },
        'numba': {
            'handlers': ['syslog'],
            'level': 'INFO',
            'propagate': False
        },
    }
}

AUTHENTICATION_BACKENDS = [
    'axes.backends.AxesBackend',
    # Default
    'django.contrib.auth.backends.ModelBackend',
]


# try to make sure this always happens before using pyplot
from dtk.plot import mpl_non_interactive
mpl_non_interactive()
