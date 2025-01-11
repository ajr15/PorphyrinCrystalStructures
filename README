# Code repository for paper: "Data-Driven Insights into Porphyrin Geometry: Interpretable AI for Non-Planarity and Aromaticity Analyses:

This repository contains all the code required to create the database, train the models and create the figures shown in the paper. 

## Setup
To setup the environment to run the code, run `source setup.bash`. Additionally, the python environment required to running is detailed in the `environment.yml` file, install using conda by running the command `conda env create -f environment.yml`. Also the Porphystruct software should be installed on your system. It can be done by following instructions on https://github.com/JensKrumsieck/PorphyStruct.

## Structure of the repo
All the code is found in the `src` directory. The function of the different scripts is detailed below
- `config.py`: Contains environment configuration values
- `utils.py`: Contains util functions
- `scrape.py`: Option to use the COD as a datasource for the project. this scripts scrapes the COD for corrole and porphyrin structures
- `cif_to_xyz.py`: Convert raw CIF files (from either CCDC or COD) to molecular XYZ files
- `curate_structure.py`: Apply the filters detailed in the paper to ensure only true porphyrins are in the dataset
- `homa_on_ideal_structures.py`: Contains the calculation of HOMA scores on ideally distorted structures
- `featurizers.py`: Contains Featurizer objects used for model training and testing
- `train_models.py`: main script to train all models used in the analysis. saves all results to `models` directory
- `read_to_sql.py`: parent script to read data on structures to the `main.db` databse file. it uses the parser functions defined in `parsers` directory. the different parsers are
    - *charge analyzer*: estimate the charge of the metal center in the complex
    - *cone angles*: calculate substituents cone angles in each structure
    - *homa scores*: calculate homa, geo and en metrics for each structure
    - *porphyrstruct*: use the Porphystruct software to run NSD calculation for each structure
    - *ring distances*: calculates the ring distances between pairs of substituents
    - *structure details*: fill in basic structure details
    - *substituents*: fill in substituent details (SMILES, atom indices, position) in each structure

In the `notebooks` directory, the `main` notebook contains all the code for the analysis presented in the paper.

## Structure of the database
`main.db` is an `sqlite` databse file containing all the information for the analysis. it contains the following tables
- *structures*: basic details (ID, file locations, SMILES) on each structure
- *substituents*: description of the substituents of each structure (position, position index, atom indicis of substituent, SMILES)
- *structure_properties*: properties calculated on the structure (HOMA, non-planarity...)
- *substituents_properties*: properties calculated on the substituents (cone angles, substituent distances...)