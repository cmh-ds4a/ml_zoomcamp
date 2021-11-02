#!/usr/bin/env python
# coding: utf-8

import requests

url = 'http://localhost:9696/predict'

home = {
 'subdivision': '751 South',
 'total_living_area_sqft': 3400,
 'year_built': 2020,
 'bedrooms': 5,
 'full_baths': 3,
 'half_baths': 0,
 'property_type': 'Detached',
 'acres': '0-.25 Acres',
 'approx_lot_sqft': 3484.8,
 'approximate_acres': 0.08,
 'basement': 'No',
 'construction_type': 'Site Built',
 'days_on_market': 68,
 'fireplace': '1',
 'garage': 2,
 'hoa_1_fees_required': 'Yes',
 'internet_listing': 'Yes',
 'master_bedroom_1st_floor': 'No',
 'new_construction': 'No',
 'total_baths': 3,
 'zip': '27713',
 'inside_city': 'No',
 'elementary_school_1': 'Durham - Creekside',
 'high_school_1': 'Durham - Jordan',
 'middle_school_1': 'Durham - Githens',
 'restrictive_covenants': 'Yes',
 'closing_month': 11,
 'closing_day': 1
}


response = requests.post(url, json=home).json()
print()

print('The estimated value for the home is ${:,.2f}'.format(response['home_price']))


