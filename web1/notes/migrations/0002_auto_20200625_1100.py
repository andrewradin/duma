# Generated by Django 2.2.10 on 2020-06-25 18:00

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('notes', '0001_squashed_0004_merge_20191016_1531'),
    ]

    operations = [
        migrations.AlterIndexTogether(
            name='noteversion',
            index_together={('created_by',)},
        ),
    ]