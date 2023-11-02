from django.db import models

# Create your models here.
class ImportAudit(models.Model):
    timestamp = models.DateTimeField(auto_now=True)
    ws = models.ForeignKey("browse.Workspace", on_delete=models.CASCADE)
    user = models.CharField(max_length=70)
    collection = models.ForeignKey("drugs.Collection", on_delete=models.CASCADE)
    operation = models.CharField(max_length=70)
    succeeded = models.BooleanField(default=False)
    clust_ver = models.IntegerField()
    coll_ver = models.IntegerField(null=True)
    # For one-off imports, we'll have a single WSA that was imported
    wsa = models.ForeignKey("browse.WsAnnotation", on_delete=models.CASCADE, null=True)

