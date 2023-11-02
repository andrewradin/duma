# -*- coding: utf-8 -*-
# Generated by Django 1.11.24 on 2020-01-10 22:59
from __future__ import unicode_literals

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    replaces = [('patsearch', '0001_initial'), ('patsearch', '0002_patentsearch_query'), ('patsearch', '0003_drugdiseasepatentsearch_wsa'), ('patsearch', '0004_bigquerypatentsearchresult'), ('patsearch', '0005_auto_20191011_1433'), ('patsearch', '0005_auto_20191011_1354'), ('patsearch', '0006_merge_20191016_1531')]

    initial = True

    dependencies = [
        ('browse', '0124_auto_20190508_1000'),
        ('runner', '0016_auto_20171114_1306'),
    ]

    operations = [
        migrations.CreateModel(
            name='DrugDiseasePatentSearch',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('query', models.TextField()),
                ('drug_name', models.CharField(max_length=255)),
            ],
        ),
        migrations.CreateModel(
            name='GooglePatentSearch',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('query', models.TextField()),
                ('total_results', models.IntegerField()),
                ('href', models.TextField()),
            ],
        ),
        migrations.CreateModel(
            name='GooglePatentSearchResult',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('search_snippet', models.TextField()),
            ],
        ),
        migrations.CreateModel(
            name='Patent',
            fields=[
                ('pub_id', models.CharField(max_length=255, primary_key=True, serialize=False)),
                ('title', models.TextField(blank=True, null=True)),
                ('abstract_snippet', models.TextField(blank=True, null=True)),
                ('date', models.DateField(blank=True, null=True)),
                ('href', models.TextField(blank=True, null=True)),
            ],
        ),
        migrations.CreateModel(
            name='PatentContentInfo',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('has_abstract', models.BooleanField()),
                ('has_claims', models.BooleanField()),
                ('job', models.ForeignKey(null=True, on_delete=django.db.models.deletion.CASCADE, to='runner.Process')),
            ],
        ),
        migrations.CreateModel(
            name='PatentFamily',
            fields=[
                ('family_id', models.CharField(max_length=32, primary_key=True, serialize=False)),
            ],
        ),
        migrations.CreateModel(
            name='PatentSearch',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('created', models.DateTimeField(auto_now_add=True)),
                ('user', models.CharField(max_length=70)),
                ('job', models.ForeignKey(null=True, on_delete=django.db.models.deletion.CASCADE, to='runner.Process')),
                ('ws', models.ForeignKey(null=True, on_delete=django.db.models.deletion.CASCADE, to='browse.Workspace')),
                ('query', models.TextField(blank=True, null=True)),
            ],
        ),
        migrations.CreateModel(
            name='PatentSearchResult',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('resolution', models.IntegerField(choices=[(0, 'Unresolved'), (1, 'Relevant'), (2, 'Irrelevant Drug'), (3, 'Irrelevant Disease'), (4, 'Irrelevant All'), (5, 'Needs More Review'), (6, 'Skipped')], default=0)),
                ('score', models.FloatField()),
                ('evidence', models.TextField()),
                ('google_patent_search_result', models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.CASCADE, to='patsearch.GooglePatentSearchResult')),
                ('patent', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='patsearch.Patent')),
                ('search', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='patsearch.DrugDiseasePatentSearch')),
            ],
        ),
        migrations.AddField(
            model_name='patentcontentinfo',
            name='patent_family',
            field=models.ForeignKey(null=True, on_delete=django.db.models.deletion.CASCADE, to='patsearch.PatentFamily'),
        ),
        migrations.AddField(
            model_name='patentcontentinfo',
            name='ws',
            field=models.ForeignKey(null=True, on_delete=django.db.models.deletion.CASCADE, to='browse.Workspace'),
        ),
        migrations.AddField(
            model_name='patent',
            name='family',
            field=models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.CASCADE, to='patsearch.PatentFamily'),
        ),
        migrations.AddField(
            model_name='googlepatentsearchresult',
            name='patent',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='patsearch.Patent'),
        ),
        migrations.AddField(
            model_name='googlepatentsearchresult',
            name='search',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='patsearch.GooglePatentSearch'),
        ),
        migrations.AddField(
            model_name='drugdiseasepatentsearch',
            name='patent_search',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='patsearch.PatentSearch'),
        ),
        migrations.AddField(
            model_name='drugdiseasepatentsearch',
            name='wsa',
            field=models.ForeignKey(null=True, on_delete=django.db.models.deletion.CASCADE, to='browse.WsAnnotation'),
        ),
        migrations.CreateModel(
            name='BigQueryPatentSearchResult',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('patent', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='patsearch.Patent')),
            ],
        ),
    ]
