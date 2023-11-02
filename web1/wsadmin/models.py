from django.db import models
from browse.models import Workspace
from tools import Enum

from .custom_dpi import CustomDpiModes


class CustomDpi(models.Model):
    ws = models.ForeignKey(Workspace, on_delete=models.CASCADE)

    name = models.CharField(max_length=256)
    base_dpi = models.CharField(max_length=256)
    prot_set = models.CharField(max_length=256)
    mode = models.IntegerField(choices=CustomDpiModes.choices())
    descr = models.TextField()
    deprecated = models.BooleanField(default=False)
    uid = models.CharField(max_length=256)

    # Tracking.
    created_on = models.DateTimeField(auto_now_add=True)
    created_by = models.CharField(max_length=256)

    # Stats.
    protset_prots = models.IntegerField()
    base_prots = models.IntegerField()
    final_prots = models.IntegerField()
    base_molecules = models.IntegerField()
    final_molecules = models.IntegerField()

    class Meta:
        index_together = [
            ['uid'],
        ]
