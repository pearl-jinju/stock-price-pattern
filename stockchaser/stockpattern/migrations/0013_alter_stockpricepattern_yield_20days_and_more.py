# Generated by Django 5.0.2 on 2024-03-06 15:20

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('stockpattern', '0012_stockpricepattern_yield_20days_and_more'),
    ]

    operations = [
        migrations.AlterField(
            model_name='stockpricepattern',
            name='yield_20days',
            field=models.TextField(),
        ),
        migrations.AlterField(
            model_name='stockpricepattern',
            name='yield_5days',
            field=models.TextField(),
        ),
        migrations.AlterField(
            model_name='stockpricepattern',
            name='yield_60days',
            field=models.TextField(),
        ),
    ]
