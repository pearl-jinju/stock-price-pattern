# Generated by Django 5.0.2 on 2024-03-07 04:47

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('stockpattern', '0014_stocknameall'),
    ]

    operations = [
        migrations.AddField(
            model_name='stockpricepattern',
            name='rate_volume_list',
            field=models.TextField(default=None),
        ),
    ]
