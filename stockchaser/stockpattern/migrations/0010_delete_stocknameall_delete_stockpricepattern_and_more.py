# Generated by Django 5.0.2 on 2024-03-06 07:34

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('stockpattern', '0009_stockpricepattern'),
    ]

    operations = [
        migrations.DeleteModel(
            name='StockNameAll',
        ),
        migrations.DeleteModel(
            name='StockPricePattern',
        ),
        migrations.RemoveField(
            model_name='stockpricedatebase',
            name='bps',
        ),
        migrations.RemoveField(
            model_name='stockpricedatebase',
            name='div',
        ),
        migrations.RemoveField(
            model_name='stockpricedatebase',
            name='dps',
        ),
        migrations.RemoveField(
            model_name='stockpricedatebase',
            name='eps',
        ),
        migrations.RemoveField(
            model_name='stockpricedatebase',
            name='pbr',
        ),
        migrations.RemoveField(
            model_name='stockpricedatebase',
            name='per',
        ),
        migrations.RemoveField(
            model_name='stockpricedatebase',
            name='volume_amount',
        ),
    ]
