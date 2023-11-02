from django.db import models

from browse.models import Prescreen, WsAnnotation

# Create your models here.

class PrescreenEntry(models.Model):
    prescreen = models.ForeignKey(Prescreen, on_delete=models.CASCADE)
    wsa = models.ForeignKey(WsAnnotation, on_delete=models.CASCADE)

    class Meta:
        unique_together = [['prescreen', 'wsa']]