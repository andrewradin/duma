# Generated by Django 2.2.10 on 2020-07-07 18:45

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('wsmgr', '0001_initial'),
    ]

    operations = [
        migrations.AddField(
            model_name='importaudit',
            name='clust_ver',
            field=models.IntegerField(default=0),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name='importaudit',
            name='coll_ver',
            field=models.IntegerField(null=True),
        ),
    ]
