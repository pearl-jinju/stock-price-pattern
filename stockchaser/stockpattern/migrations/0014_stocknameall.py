# Generated by Django 5.0.2 on 2024-03-06 15:56

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('stockpattern', '0013_alter_stockpricepattern_yield_20days_and_more'),
    ]

    operations = [
        migrations.CreateModel(
            name='StockNameAll',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.TextField()),
                ('ticker', models.TextField()),
            ],
        ),
    ]