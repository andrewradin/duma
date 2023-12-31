# -*- coding: utf-8 -*-
# Generated by Django 1.11.27 on 2020-01-10 23:39
from __future__ import unicode_literals

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    replaces = [('ktsearch', '0001_initial'), ('ktsearch', '0002_ktresultgroup_resolution'), ('ktsearch', '0003_auto_20190319_1102'), ('ktsearch', '0004_ktsearchresult_unmatch_confirmed'), ('ktsearch', '0005_auto_20190424_1620'), ('ktsearch', '0006_auto_20190508_1000'), ('ktsearch', '0007_auto_20190808_1447'), ('ktsearch', '0008_auto_20190918_1636'), ('ktsearch', '0009_auto_20191011_1433'), ('ktsearch', '0009_auto_20191011_1354'), ('ktsearch', '0010_merge_20191016_1531'), ('ktsearch', '0011_auto_20191022_1715')]

    initial = True

    dependencies = [
        ('browse', '0115_wsannotation_review_code'),
    ]

    operations = [
        migrations.CreateModel(
            name='KtResultGroup',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(default=b'', max_length=256)),
                ('wsa', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='browse.WsAnnotation')),
            ],
        ),
        migrations.CreateModel(
            name='KtSearch',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('created', models.DateTimeField(auto_now_add=True)),
                ('user', models.CharField(max_length=70)),
                ('ws', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='browse.Workspace')),
            ],
        ),
        migrations.CreateModel(
            name='KtSearchResult',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('drugname', models.CharField(default=b'', max_length=256)),
                ('href', models.CharField(default=b'', max_length=1024)),
                ('extra', models.TextField(blank=True, default=b'')),
                ('ind_val', models.IntegerField(choices=[(0, b'Unclassified'), (1, b'FDA Approved Treatment'), (2, b'FDA Documented Cause'), (3, b'Clinically used treatment'), (4, b'Clinically indicated cause'), (5, b'Candidate Treatment'), (6, b'Candidate Cause'), (7, b'Researched as treatment'), (8, b'Researched as cause'), (9, b'Inactive Candidate'), (10, b'Patent submitted'), (11, b'Preparing patent'), (12, b'Clinically investigated treatment'), (13, b'Hypothesized treatment')])),
                ('group', models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.CASCADE, to='ktsearch.KtResultGroup')),
            ],
        ),
        migrations.CreateModel(
            name='KtSource',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('source_type', models.CharField(max_length=70)),
                ('config', models.TextField(blank=True, default='')),
                ('search', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='ktsearch.KtSearch')),
            ],
        ),
        migrations.AddField(
            model_name='ktsearchresult',
            name='query',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='ktsearch.KtSource'),
        ),
        migrations.AddField(
            model_name='ktresultgroup',
            name='resolution',
            field=models.IntegerField(choices=[(0, 'Unresolved'), (1, 'Accepted'), (2, 'Skipped'), (3, 'Matched Existing')], default=0),
        ),
        migrations.AlterField(
            model_name='ktsearchresult',
            name='ind_val',
            field=models.IntegerField(choices=[(0, b'Unclassified'), (1, b'FDA Approved Treatment'), (2, b'FDA Documented Cause'), (3, b'Clinically used treatment'), (4, b'Clinically indicated cause'), (5, b'Candidate Treatment'), (6, b'Candidate Cause'), (7, b'Researched as treatment'), (8, b'Researched as cause'), (9, b'Inactive Candidate'), (10, b'Patent submitted'), (11, b'Preparing patent'), (12, b'Clinically investigated treatment'), (13, b'Hypothesized treatment'), (14, b'Reviewed Candidate'), (15, b'Preclinical Candidate')]),
        ),
        migrations.AddField(
            model_name='ktsearchresult',
            name='unmatch_confirmed',
            field=models.BooleanField(default=False),
        ),
        migrations.RemoveField(
            model_name='ktresultgroup',
            name='name',
        ),
        migrations.AddField(
            model_name='ktresultgroup',
            name='search',
            field=models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.CASCADE, to='ktsearch.KtSearch'),
        ),
        migrations.AlterField(
            model_name='ktsearchresult',
            name='ind_val',
            field=models.IntegerField(choices=[(0, b'Unclassified'), (1, b'FDA Approved Treatment'), (2, b'FDA Documented Cause'), (3, b'Clinically used treatment'), (4, b'Clinically indicated cause'), (5, b'Candidate Treatment'), (6, b'Candidate Cause'), (7, b'Researched as treatment'), (8, b'Researched as cause'), (9, b'Inactive Candidate'), (10, b'Patent submitted'), (11, b'Preparing patent'), (12, b'Clinically investigated treatment'), (13, b'Hypothesized treatment'), (14, b'Reviewed Candidate'), (15, b'Preclinical Candidate'), (16, b'Phase 1 treatment'), (17, b'Phase 2 treatment'), (18, b'Phase 3 treatment')]),
        ),
        migrations.AddField(
            model_name='ktresultgroup',
            name='timestamp',
            field=models.DateTimeField(null=True),
        ),
        migrations.AddField(
            model_name='ktresultgroup',
            name='user',
            field=models.CharField(blank=True, default='', max_length=50),
        ),
        migrations.AlterField(
            model_name='ktsearchresult',
            name='ind_val',
            field=models.IntegerField(choices=[(0, b'Unclassified'), (1, b'FDA Approved Treatment'), (2, b'FDA Documented Cause'), (3, b'Clinically used treatment'), (4, b'Clinically indicated cause'), (5, b'Candidate Treatment'), (6, b'Candidate Cause'), (7, b'Researched as treatment'), (8, b'Researched as cause'), (9, b'Inactive Candidate'), (10, b'Patent submitted'), (11, b'Preparing patent'), (12, b'Clinically investigated treatment'), (13, b'Hypothesized treatment'), (14, b'Reviewed Candidate'), (15, b'Preclinical Candidate'), (16, b'Phase 1 treatment'), (17, b'Phase 2 treatment'), (18, b'Phase 3 treatment'), (19, b'In Vitro 1'), (20, b'In Vitro 2'), (21, b'In Vivo 1'), (22, b'In Vivo 2')]),
        ),
        migrations.AlterField(
            model_name='ktsearchresult',
            name='drugname',
            field=models.CharField(default='', max_length=256),
        ),
        migrations.AlterField(
            model_name='ktsearchresult',
            name='extra',
            field=models.TextField(blank=True, default=''),
        ),
        migrations.AlterField(
            model_name='ktsearchresult',
            name='href',
            field=models.CharField(default='', max_length=1024),
        ),
        migrations.AlterField(
            model_name='ktsearchresult',
            name='ind_val',
            field=models.IntegerField(choices=[(0, 'Unclassified'), (1, 'FDA Approved Treatment'), (2, 'FDA Documented Cause'), (3, 'Clinically used treatment'), (4, 'Clinically indicated cause'), (5, 'Candidate Treatment'), (6, 'Candidate Cause'), (7, 'Researched as treatment'), (8, 'Researched as cause'), (9, 'Inactive Candidate'), (10, 'Patent submitted'), (11, 'Preparing patent'), (12, 'Clinically investigated treatment'), (13, 'Hypothesized treatment'), (14, 'Reviewed Candidate'), (15, 'Preclinical Candidate'), (16, 'Phase 1 treatment'), (17, 'Phase 2 treatment'), (18, 'Phase 3 treatment'), (19, 'In Vitro 1'), (20, 'In Vitro 2'), (21, 'In Vivo 1'), (22, 'In Vivo 2')]),
        ),
        migrations.AlterField(
            model_name='ktsearchresult',
            name='drugname',
            field=models.CharField(default='', max_length=256),
        ),
        migrations.AlterField(
            model_name='ktsearchresult',
            name='extra',
            field=models.TextField(blank=True, default=''),
        ),
        migrations.AlterField(
            model_name='ktsearchresult',
            name='href',
            field=models.CharField(default='', max_length=1024),
        ),
        migrations.AlterField(
            model_name='ktsearchresult',
            name='ind_val',
            field=models.IntegerField(choices=[(0, 'Unclassified'), (1, 'FDA Approved Treatment'), (2, 'FDA Documented Cause'), (3, 'Clinically used treatment'), (4, 'Clinically indicated cause'), (5, 'Candidate Treatment'), (6, 'Candidate Cause'), (7, 'Researched as treatment'), (8, 'Researched as cause'), (9, 'Inactive Candidate'), (10, 'Patent submitted'), (11, 'Preparing patent'), (12, 'Clinically investigated treatment'), (13, 'Hypothesized treatment'), (14, 'Reviewed Candidate'), (15, 'Preclinical Candidate'), (16, 'Phase 1 treatment'), (17, 'Phase 2 treatment'), (18, 'Phase 3 treatment'), (19, 'In Vitro 1'), (20, 'In Vitro 2'), (21, 'In Vivo 1'), (22, 'In Vivo 2')]),
        ),
        migrations.AlterField(
            model_name='ktsearchresult',
            name='ind_val',
            field=models.IntegerField(choices=[(0, 'Unclassified'), (1, 'FDA Approved Treatment'), (2, 'FDA Documented Cause'), (3, 'Clinically used treatment'), (4, 'Clinically indicated cause'), (5, 'Initial Prediction'), (6, 'Candidate Cause'), (7, 'Researched as treatment'), (8, 'Researched as cause'), (9, 'Inactive Prediction'), (10, 'Patent submitted'), (11, 'Preparing patent'), (12, 'Clinically investigated treatment'), (13, 'Hypothesized treatment'), (14, 'Reviewed Prediction'), (15, 'Hit'), (16, 'Phase 1 treatment'), (17, 'Phase 2 treatment'), (18, 'Phase 3 treatment'), (19, 'In Vitro 1'), (20, 'In Vitro 2'), (21, 'In Vivo 1'), (22, 'In Vivo 2')]),
        ),
    ]
