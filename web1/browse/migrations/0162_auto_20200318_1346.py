# Generated by Django 2.2.10 on 2020-03-18 20:46

from django.db import migrations, models

def set_dates(apps, schema_editor):
    from datetime import datetime
    import pytz
    start = datetime.fromtimestamp(0)
    start = start.replace(tzinfo=pytz.UTC)

    ProtSet = apps.get_model('browse', 'ProtSet')
    for ps in ProtSet.objects.all():
        ps.created_on = start
        ps.save()

    DrugSet = apps.get_model('browse', 'DrugSet')
    for ds in DrugSet.objects.all():
        ds.created_on = start
        ds.save()


class Migration(migrations.Migration):

    dependencies = [
        ('browse', '0161_aeaccession_alt_ids'),
    ]

    operations = [
        migrations.AddField(
            model_name='drugset',
            name='created_by',
            field=models.CharField(default='', max_length=256),
        ),
        migrations.AddField(
            model_name='drugset',
            name='created_on',
            field=models.DateTimeField(auto_now_add=True, null=True),
        ),
        migrations.AddField(
            model_name='drugset',
            name='description',
            field=models.TextField(default=''),
        ),
        migrations.AddField(
            model_name='protset',
            name='created_by',
            field=models.CharField(default='', max_length=256),
        ),
        migrations.AddField(
            model_name='protset',
            name='created_on',
            field=models.DateTimeField(auto_now_add=True, null=True),
        ),
        migrations.AddField(
            model_name='protset',
            name='description',
            field=models.TextField(default=''),
        ),
        migrations.RunPython(
            set_dates,
            reverse_code=lambda *args, **kwargs: None),
    ]