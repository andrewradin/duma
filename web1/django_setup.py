import os
import sys
if ("DJANGO_SETTINGS_MODULE" not in os.environ or
    "django.core" not in sys.modules):
    import django
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "web1.settings") 
    django.setup() 
