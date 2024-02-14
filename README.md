# VOC-EnthVapML
This is the repository of the paper "Data-Driven, Explainable Machine Learning Model for Predicting Volatile Organic Compounds’ Standard Vaporization Enthalpy"




# File Overview

Database/

Mode_Database.csv: Complete dataset with descriptor calculations

ML_Vap_Full_Database_v1.xlsx: Raw complete dataset with internal and exteranal sources


Scripts/

rdkit_conversion_vap.py: Calculation of desired RdKit descriptors using the raw database

model_vap_calculations.py: Model calculations using desired algorithms with all calculated descriptors

model_desc_groups.py: Model performance using only best descriptors for model optimization

permut_importance_vap.py: Routine to determine best descriptors using permuataion importance

model_holdout_tests.py: Routine to perform chemical familiy holdout tests using best descriptors

append_NIST.py: Script to extract enthalpy values from NIST (Forked from vmurc - victormurcia @https://github.com/victormurcia/nist_webscraper)


Results/

ML_Vap_Full_Results_SI.xlsx: File including all model results presented in the paper (including permuation importance, model statistical performance, solvent holdout tests and descriptors group performance determinations
