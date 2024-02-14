# VOC-EnthVapML
This is the repository of the paper "Data-Driven, Explainable Machine Learning Model for Predicting Volatile Organic Compounds’ Standard Vaporization Enthalpy"




# File Overview

Database/

VOC-Database.csv: Complete dataset with descriptor calculations to run the model

VOC-Database-Global.xlsx: Raw complete dataset with internal and external sources

desc_VOC-RF.csv: Descriptor group categorisation


Scripts/

rdkit_conversion_vap.py: Calculation of desired RdKit descriptors using the raw database

dVap-Model-Testing.py: Model calculations using desired algorithms with all calculated descriptors

VOC-Model-Testing.py: Model calculations for VOC with all calculated descriptors and hyperparameter optimization

dVap-Family-Group-Testing.py: Model calculations for VOC descriptor groups and chemical family testing



Results/

VOC-ML-Full-Results.xlsx: File including all model results presented in the paper (including permutation importance, model statistical performance and descriptors group performance determinations)
