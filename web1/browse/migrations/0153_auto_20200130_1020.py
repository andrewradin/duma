# -*- coding: utf-8 -*-
# Generated by Django 1.11.27 on 2020-01-30 18:20
from __future__ import unicode_literals

from django.db import migrations


def func(apps, schema_editor):
    # Copy all the statuses from 
    StageStatus = apps.get_model('browse', 'StageStatus')
    ge_statuses = StageStatus.objects.filter(stage_name='GeneExpressionData')

    for stat in ge_statuses:
        obj, is_new = StageStatus.objects.get_or_create(
                ws=stat.ws,
                stage_name='SearchTissueStage'
                )
        obj.status = stat.status
        obj.save()
    




class Migration(migrations.Migration):

    dependencies = [
        ('browse', '0152_auto_20200127_1249'),
    ]

    operations = [
            migrations.RunPython(func)
    ]
