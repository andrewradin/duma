# Generated by Django 2.2.10 on 2020-04-23 19:46

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('drugs', '0005_auto_20200323_1316'),
    ]

    operations = [
        migrations.AlterIndexTogether(
            name='dpimergekey',
            index_together={('version', 'dpimerge_key'), ('version', 'drug')},
        ),
    ]