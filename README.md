# AML-Node16-challenge
Decentralized AI for anti-money laundering (AML)

Download root data from 
https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml/data?select=HI-Large_Trans.csv


The raw data has been cleaned and preprocessed through the "EDA & Preprocessing" Notebook and clean data has been stored in parquet format on the clean_data folder.

# How to Run:
# Git clone the repository

``` bash
Project repo: https://github.com/sm-ak-r33/AML-Node16-challenge.git
```
# Create an environment
```bash 
conda create -n newenv python=3.9 -y
```

```bash
conda activate newenv
```

# Install the requirements
```bash
pip install -r requirements.txt 
``` 

# To Run locally 
```bash
python run_AML.py
```
