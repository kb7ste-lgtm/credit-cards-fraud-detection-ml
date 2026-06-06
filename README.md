# credit-cards-fraud-detection-ml
Il s'agit d'une application streamlit dédiée à la détection de fraudes par carte bancaire.
## Description
Ce projet repose sur un modèle LightGBM pour identifier les transactions frauduleuse sur des variables ayant subit un PCA au préalable sauf deux qui ont été scalé. L'interface utilisateur permet à l'utilisateur d'insérer son input de deux manières : via des barres de scaling et le traitement /upload de fichier CSV.
## Overview des fonctionnalités : 
-** interface interractive via les sliders pour tester differentes variables PCA(V1 à V28).
-**upload de fichier csv des transactions : ** possibilité d'uploader un fichier csv pour analyser des transactions.
-** Design simple et friendly.
## librairies utilisées :  Streamlit, Pandas, numpy, Scikit-learn (RobustScaler), LightGBM, Imbalanced-learn
##Auteur
Projet réalisé dans le cadre de mon master à l'EHTP.
