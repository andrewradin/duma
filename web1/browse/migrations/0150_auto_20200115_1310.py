# -*- coding: utf-8 -*-
# Generated by Django 1.11.23 on 2020-01-15 21:10
from __future__ import unicode_literals

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('browse', '0149_wsannotation_replaced_by'),
    ]

    operations = [
        migrations.AlterField(
            model_name='runset',
            name='ws',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='browse.Workspace'),
        ),
        migrations.AlterField(
            model_name='sample',
            name='attributes',
            field=models.TextField(blank=True),
        ),
    ]
