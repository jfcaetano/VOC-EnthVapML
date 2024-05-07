# VOC-EnthVapML
This is the repository for the research project: "Data-Driven, Explainable Machine Learning Model for Predicting Volatile Organic Compounds’ Standard Vaporization Enthalpy" DOI: https://doi.org/10.1016/j.chemosphere.2024.142257
The complete dataset is doposited in Zenodo: https://zenodo.org/doi/10.5281/zenodo.11127879

# File Overview

Database

VOC-Database.csv: Complete dataset with descriptor calculations to run the model

Database-Global.xlsx: Raw complete dataset with internal and external sources

desc_VOC-RF.csv: Descriptor group categorisation


Scripts

rdkit_conversion_vap.py: Calculation of desired RDKit descriptors using the raw database

dVap-Model-Testing.py: Model calculations using desired algorithms with all calculated descriptors

VOC-Model-Testing.py: Model calculations for VOC with all calculated descriptors and hyperparameter optimization

dVap-Family-Group-Testing.py: Model calculations for VOC descriptor groups and chemical family testing



Results/

VOC-ML-Full-Results.xlsx: File including all model results presented in the paper (including permutation importance, model statistical performance and descriptors group performance determinations)

dVap-ML-Dev-Results.xlsx: File including model results for preiction optimization of the standard enthalpy of vaporization (including permutation importance, model statistical performance and descriptors group performance determinations).
