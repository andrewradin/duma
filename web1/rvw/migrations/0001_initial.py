# Generated by Django 2.2.16 on 2020-09-25 20:27

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    initial = True

    dependencies = [
        ('browse', '0168_auto_20200925_1327'),
    ]

    operations = [
        migrations.CreateModel(
            name='PrescreenEntry',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('prescreen', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='browse.Prescreen')),
                ('wsa', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='browse.WsAnnotation')),
            ],
            options={
                'unique_together': {('prescreen', 'wsa')},
            },
        ),
    ]
