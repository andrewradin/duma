# Generated by Django 2.2.19 on 2021-03-31 16:49

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('browse', '0174_aeaccession_num_samples'),
    ]

    operations = [
        migrations.AlterIndexTogether(
            name='aeaccession',
            index_together={('geoID',)},
        ),
    ]
