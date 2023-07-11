# EnthVapML
This is the repository of the JPC paper "Machine Learning Model to Predict Standard Vaporization Enthalpies through Intelligible Features"
The full database is available at Zenodo (https://doi.org/10.5281/zenodo.8132046)


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



Results/

ML_Vap_Full_Results_SI.xlsx: File including all model results presented in the paper (including permuation importance, model statistical performance, solvent holdout tests and descriptors group performance determinations

# Authorship
Code was written by José Ferraz-Caetano, under the supervision of Filipe Teixeira and Natália Cordeiro.

# Acknowledgements
This code was developed at the Univerisity of Porto and was supported by the "Fundação para a Ciência e Tecnologia" (FCT/MCTES) to LAQV-REQUIMTE Lab (UIDP/50006/2020). JFC’s PhD Fellowship is supported by the doctoral Grant (SFRH/BD/151159/2021) financed by FCT, with funds from the Portuguese State and EU Budget through the Social European Fund and Programa Por_Centro, under the MIT Portugal Program.

# BibTex
