# Generated by Django 2.2.10 on 2020-05-31 21:13

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('drugs', '0006_auto_20200423_1246'),
    ]

    operations = [
        migrations.AlterIndexTogether(
            name='tag',
            index_together={('value',), ('prop', 'value')},
        ),
    ]