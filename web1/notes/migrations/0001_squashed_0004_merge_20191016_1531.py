# -*- coding: utf-8 -*-
# Generated by Django 1.11.27 on 2020-01-10 23:35
from __future__ import unicode_literals

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    replaces = [('notes', '0001_initial'), ('notes', '0002_auto_20160204_1314'), ('notes', '0003_auto_20191011_1433'), ('notes', '0003_auto_20191011_1354'), ('notes', '0004_merge_20191016_1531')]

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Note',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
            ],
        ),
        migrations.CreateModel(
            name='NoteVersion',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('text', models.TextField()),
                ('created_by', models.CharField(blank=True, default='', max_length=50)),
                ('created_on', models.DateTimeField(auto_now_add=True)),
                ('version_of', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='notes.Note')),
            ],
        ),
        migrations.AddField(
            model_name='note',
            name='label',
            field=models.CharField(blank=True, default='', max_length=250),
        ),
        migrations.AddField(
            model_name='note',
            name='private_to',
            field=models.CharField(blank=True, default='', max_length=50),
        ),
    ]