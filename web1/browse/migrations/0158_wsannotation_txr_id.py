# -*- coding: utf-8 -*-
# Generated by Django 1.11.27 on 2020-02-06 17:47
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('browse', '0157_merge_20200205_1046'),
    ]

    operations = [
        migrations.AddField(
            model_name='wsannotation',
            name='txr_id',
            field=models.CharField(blank=True, default='', max_length=50),
        ),
    ]
