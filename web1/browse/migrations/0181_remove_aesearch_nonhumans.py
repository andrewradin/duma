# Generated by Django 2.2.24 on 2022-03-15 21:24

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('browse', '0180_auto_20220315_1404'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='aesearch',
            name='nonhumans',
        ),
    ]
