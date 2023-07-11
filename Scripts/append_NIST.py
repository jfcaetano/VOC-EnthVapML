### Webscrapper NIST ΔvapH° experimental values
### JFCAETANO 2023
### MIT Licence
### Forked from vmurc - victormurcia @https://github.com/victormurcia/nist_webscraper

import pandas as pd
import numpy as np
from requests import get
from bs4 import BeautifulSoup
import csv

def get_thermo_info(material):
    mat_corr = material
    url = f"https://webbook.nist.gov/cgi/cbook.cgi?InChI={mat_corr}&Units=SI&cTP=on"
    raw_html = requests.get(url).text
    soup = BeautifulSoup(raw_html, 'html.parser')
    df = pd.DataFrame(columns =['0', '1', '2', '3', '4', '5'])
    try:
        table = soup.find('table', attrs={'aria-label': 'One dimensional data'})
    except: 
        df = df.append({'1': np.nan,'2': np.nan,'3': np.nan,'4': np.nan, '5': np.nan},ignore_index=True)
        return df
    try:
        rows = table.findAll('tr')
    except:
        df = df.append({'1': np.nan,'2': np.nan,'3': np.nan,'4': np.nan, '5': np.nan},ignore_index=True)
        return df

    data = []
    for row in rows:
        cols = row.find_all('td')
        cols = [x.text.strip() for x in cols]
        if len(cols) > 0:
            data.append(cols)
            
    df = pd.DataFrame(data)
    df = pd.DataFrame(data, columns =['0', '1', '2', '3', '4', '5'])
    df = df[(df['0'] == 'ΔvapH°')]
    df = df.iloc[:1]
    
    if df.empty:
        df = df.append({'0': np.nan,'1': np.nan,'2': np.nan,'3': np.nan,'4': np.nan, '5': np.nan},ignore_index=True)
        
    return df

appended_data = []
for item in y:
    df = get_thermo_info(item)
    # store DataFrame in list
    appended_data.append(df)
# see pd.concat documentation for more info
appended_data = pd.concat(appended_data)
# write DataFrame to an excel sheet 
appended_data.to_csv('data.csv')
