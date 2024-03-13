# Generated by Django 2.2.24 on 2022-05-18 21:58

from django.db import migrations

def patch_roles(apps, schema_editor):
    Process = apps.get_model('runner','Process')
    for p in Process.objects.filter(role__startswith='dgn_'):
        p.role = 'dgns_'+p.role
        p.save()
    for p in Process.objects.filter(role__startswith='agr_'):
        p.role = 'agrs_'+p.role
        p.save()
    for p in Process.objects.filter(role__startswith='misig_'):
        if p.role.startswith('misig_misig_'):
            continue
        p.role = 'misig_'+p.role
        p.save()

class Migration(migrations.Migration):

    dependencies = [
        ('runner', '0003_processwait'),
    ]

    operations = [
        migrations.RunPython(patch_roles),
    ]