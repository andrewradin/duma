# -*- coding: utf-8 -*-
# Generated by Django 1.11.24 on 2020-01-10 22:49
from __future__ import unicode_literals

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    replaces = [('drugs', '0001_initial'), ('drugs', '0002_auto_20150421_0934'), ('drugs', '0003_prop_multival'), ('drugs', '0004_auto_20150421_1141'), ('drugs', '0005_index_metric'), ('drugs', '0006_remove_drug_name'), ('drugs', '0007_drug_note'), ('drugs', '0008_auto_20150514_2224'), ('drugs', '0009_drug_hide'), ('drugs', '0010_collection_key_name'), ('drugs', '0011_drug_ubiquitous'), ('drugs', '0012_drug_removed'), ('drugs', '0013_uploadaudit'), ('drugs', '0014_remove_uploadaudit_collection'), ('drugs', '0015_remove_drug_note'), ('drugs', '0016_drug_bd_note'), ('drugs', '0017_drugproposal'), ('drugs', '0018_drugproposal_drug_name'), ('drugs', '0019_auto_20190916_1357'), ('drugs', '0020_auto_20190916_1532'), ('drugs', '0021_auto_20191001_1516'), ('drugs', '0022_auto_20191011_1433'), ('drugs', '0022_auto_20191011_1354'), ('drugs', '0023_merge_20191016_1531')]

    initial = True

    dependencies = [
        ('notes', '0001_initial'),
        ('notes', '0002_auto_20160204_1314'),
    ]

    operations = [
        migrations.CreateModel(
            name='Collection',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(default=b'', max_length=256)),
            ],
        ),
        migrations.CreateModel(
            name='Drug',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(default=b'', max_length=256)),
                ('collection', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='drugs.Collection')),
            ],
        ),
        migrations.CreateModel(
            name='PropName',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(default=b'', max_length=256)),
                ('prop_type', models.IntegerField(choices=[(0, b'Tag'), (1, b'Flag'), (2, b'Index'), (3, b'Metric')])),
            ],
        ),
        migrations.CreateModel(
            name='Tag',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('value', models.CharField(default=b'', max_length=256)),
                ('drug', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='drugs.Drug')),
                ('prop', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='drugs.PropName')),
                ('href', models.CharField(default='', max_length=1024)),
            ],
        ),
        migrations.RenameModel(
            old_name='PropName',
            new_name='Prop',
        ),
        migrations.AddField(
            model_name='prop',
            name='multival',
            field=models.BooleanField(default=False),
        ),
        migrations.CreateModel(
            name='Flag',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('value', models.BooleanField(default=False)),
                ('href', models.CharField(default='', max_length=1024)),
                ('drug', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='drugs.Drug')),
                ('prop', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='drugs.Prop')),
            ],
        ),
        migrations.CreateModel(
            name='Index',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('value', models.IntegerField()),
                ('href', models.CharField(default='', max_length=1024)),
                ('drug', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='drugs.Drug')),
                ('prop', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='drugs.Prop')),
            ],
        ),
        migrations.CreateModel(
            name='Metric',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('value', models.FloatField()),
                ('href', models.CharField(default='', max_length=1024)),
                ('drug', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='drugs.Drug')),
                ('prop', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='drugs.Prop')),
            ],
        ),
        migrations.RemoveField(
            model_name='drug',
            name='name',
        ),
        migrations.AddField(
            model_name='drug',
            name='note',
            field=models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.CASCADE, to='notes.Note'),
        ),
        migrations.CreateModel(
            name='Blob',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('value', models.TextField()),
                ('href', models.CharField(default='', max_length=1024)),
                ('drug', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='drugs.Drug')),
                ('prop', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='drugs.Prop')),
            ],
        ),
        migrations.AlterField(
            model_name='prop',
            name='prop_type',
            field=models.IntegerField(choices=[(0, b'Tag'), (1, b'Flag'), (2, b'Index'), (3, b'Metric'), (4, b'Blob')]),
        ),
        migrations.AddField(
            model_name='drug',
            name='hide',
            field=models.BooleanField(default=False),
        ),
        migrations.AddField(
            model_name='collection',
            name='key_name',
            field=models.CharField(default='', max_length=256),
        ),
        migrations.AddField(
            model_name='drug',
            name='ubiquitous',
            field=models.BooleanField(default=False),
        ),
        migrations.AddField(
            model_name='drug',
            name='removed',
            field=models.BooleanField(default=False),
        ),
        migrations.CreateModel(
            name='UploadAudit',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('timestamp', models.DateTimeField(auto_now=True)),
                ('filename', models.CharField(max_length=256)),
                ('ok', models.BooleanField(default=False)),
            ],
        ),
        migrations.RemoveField(
            model_name='drug',
            name='note',
        ),
        migrations.AddField(
            model_name='drug',
            name='bd_note',
            field=models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.CASCADE, to='notes.Note'),
        ),
        migrations.CreateModel(
            name='DrugProposal',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('data', models.TextField()),
                ('user', models.CharField(max_length=50)),
                ('timestamp', models.DateTimeField(auto_now=True)),
                ('state', models.IntegerField(choices=[(0, 'Proposed'), (1, 'Rejected'), (2, 'Accepted'), (3, 'Skipped'), (4, 'Out Of Date')], default=0)),
                ('ref_drug', models.ForeignKey(null=True, on_delete=django.db.models.deletion.CASCADE, to='drugs.Drug')),
                ('drug_name', models.TextField(default='')),
                ('collection_drug_id', models.CharField(blank=True, max_length=16, null=True)),
                ('ref_proposal', models.ForeignKey(null=True, on_delete=django.db.models.deletion.CASCADE, to='drugs.DrugProposal')),
            ],
        ),
        migrations.AlterIndexTogether(
            name='tag',
            index_together=set([('value',)]),
        ),
        migrations.AlterField(
            model_name='collection',
            name='name',
            field=models.CharField(default='', max_length=256),
        ),
        migrations.AlterField(
            model_name='prop',
            name='name',
            field=models.CharField(default='', max_length=256),
        ),
        migrations.AlterField(
            model_name='prop',
            name='prop_type',
            field=models.IntegerField(choices=[(0, 'Tag'), (1, 'Flag'), (2, 'Index'), (3, 'Metric'), (4, 'Blob')]),
        ),
        migrations.AlterField(
            model_name='tag',
            name='value',
            field=models.CharField(default='', max_length=256),
        ),
        migrations.AlterField(
            model_name='collection',
            name='name',
            field=models.CharField(default='', max_length=256),
        ),
        migrations.AlterField(
            model_name='prop',
            name='name',
            field=models.CharField(default='', max_length=256),
        ),
        migrations.AlterField(
            model_name='prop',
            name='prop_type',
            field=models.IntegerField(choices=[(0, 'Tag'), (1, 'Flag'), (2, 'Index'), (3, 'Metric'), (4, 'Blob')]),
        ),
        migrations.AlterField(
            model_name='tag',
            name='href',
            field=models.CharField(default='', max_length=1024),
        ),
        migrations.AlterField(
            model_name='tag',
            name='value',
            field=models.CharField(default='', max_length=256),
        ),
    ]
