from django.db import models

from tools import Enum

# Create your models here.
class CtSearch(models.Model):
    ws = models.ForeignKey("browse.Workspace", on_delete=models.CASCADE)
    created = models.DateTimeField(auto_now_add=True)
    user = models.CharField(max_length=70)
    config = models.TextField(default='',blank=True)
    def remaining(self):
        return "%d of %d unresolved" % (
                self.ctdrugname_set.filter(status=0).count(),
                self.ctdrugname_set.count(),
                )
    def description(self):
        import json
        d = json.loads(self.config)
        items = [
                d['disease'],
                '(%s)'%(' '.join(d['phases'])),
                ]
        if d['completed']:
            items.append('completed only')
        if d['after']:
            items.append('after %d'%d['after'])
        return '; '.join(items)

class CtDrugName(models.Model):
    status_vals = Enum([], [
            ('UNRESOLVED',),
            ('UNMATCHED',),
            ('AMBIGUOUS',),
            ('PREMARKED',),
            ('ASSIGNED',),
            ('REJECTED',),
            ('MANUAL',),
            ])
    search = models.ForeignKey(CtSearch, on_delete=models.CASCADE)
    drug_name = models.CharField(max_length=256)
    status = models.IntegerField(
            choices=status_vals.choices(),
            default=0,
            )
    study_id = models.CharField(max_length=70,
            default='',
            blank=True,
            )

