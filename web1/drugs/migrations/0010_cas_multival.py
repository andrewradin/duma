# Generated by Django 2.2.24 on 2021-12-03 01:25

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('drugs', '0009_auto_20201119_2014'),
    ]

    operations = [
        migrations.RunSQL("update drugs_prop set multival=1 where name='cas'"),
    ]