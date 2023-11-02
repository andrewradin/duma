
from web1.settings import *

# Don't want to pollute our real (dev) caches with test data
# Also, caches can cause tests to interfere with each other.
CACHES = {
    'default': {
        'BACKEND': 'django.core.cache.backends.dummy.DummyCache'
    },
    'browse.utils.Scorebox': {
        'BACKEND': 'django.core.cache.backends.dummy.DummyCache'
    },
    'drugnotes': {
        'BACKEND': 'django.core.cache.backends.dummy.DummyCache'
    },
    'selectability': {
        'BACKEND': 'django.core.cache.backends.dummy.DummyCache'
    },
    'cached_dict_elements': {
        'BACKEND': 'django.core.cache.backends.dummy.DummyCache'
    },
}

# Remove axes, it is not pytest compatible
# See https://github.com/jazzband/django-axes/issues/340
AUTHENTICATION_BACKENDS = [
    'django.contrib.auth.backends.ModelBackend',
]