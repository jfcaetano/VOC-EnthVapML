#RDKIT CONVERSION
#JF Caetano July 2023
#MIT Licence

import rdkit, sys, time, csv
from rdkit import Chem
from rdkit.Chem import Descriptors
import pandas as pd
import numpy as np


# Setup dataframe
data_filename = 'Model_Database.csv'
output_fn = 'Model_Database_xx.csv'


my_descriptors = list()
for desc_name in dir(Descriptors):
    if desc_name in ['BalabanJ','BertzCT']: # Add other descriptor names
        my_descriptors.append(desc_name)


# Prepare calculations
f = open(data_filename,'r')
reader = csv.DictReader(f, delimiter=',')
#
o = list()
for row in reader:
    # Columns to maintain; Edit SI data file accordingly
    nl = dict()
    nl['Type']                          = row['Type']
    nl['SMILES']                        = row['SMILES']
    nl['External Database']             = row['External Database'] #presence in external database
    nl['dvap']                          = row['dvap'] #value of vaporization enthalpy  

    # Load Compound
    comp_fn = row['SMILES']
    comp_mol = Chem.MolFromSmiles(comp_fn, sanitize=True)
    # Calculate Catal Descriptors
    for desc in my_descriptors:
        nl[f"{desc}"]=eval(f"Descriptors.{desc}(comp_mol)")      
    # Append nl to output list
    
    o.append(nl)

with open(output_fn,'w',newline='') as fout:
    writer = csv.DictWriter(fout, fieldnames=o[0].keys())
    writer.writeheader()
    for new_row in o:
        writer.writerow(new_row)

# Clean up stuff
f.close()

