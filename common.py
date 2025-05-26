import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import os

def load_data (name: str) -> pd.DataFrame: # DONE
    """
    Function that loads dataset from csv file

    :param name: Name of the file
    :return: pandas dataframe where every row represents single article with attributes: title, text, subject, date
    """
    class_name = name.removesuffix(".csv")
    data = pd.read_csv(os.path.join("Datasets", name))
    data['target'] = class_name
    return data

def choose_random_part (data: pd.DataFrame, n: int) -> pd.DataFrame: # DONE
    """
    Function that randomly chooses n articles from the whole dataset

    :param data: pandas dataframe containing whole dataset
    :param n: Number of articles to be randomly sampled
    :return: Pandas Dataframe containing n random articles
    """
    return data.sample(n)

def divide_data_into_train_and_test (data: pd.DataFrame, t_size: float) -> tuple: # DONE
    """
    Function that splits dataset into train and test datasets

    :param data: pandas dataframe containing data
    :param t_size: Size of test dataset
    :return: tuple containing X_train, X_test, y_train and y_test
    """
    X = data.drop(columns=["target"])
    y = data[["target"]]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=t_size)
    return X_train, X_test, y_train, y_test

def merge_datasets(dataset1: pd.DataFrame, dataset2: pd.DataFrame) -> pd.DataFrame: # DONE
    """
    Function that merges two datasets in form of pandas dataframe into one

    :param dataset1: pandas dataframe containing first dataset
    :param dataset2: pandas dataframe containing second dataset
    :return: pandas dataframe containing merged datasets
    """
    return pd.concat([dataset1, dataset2], ignore_index=True)

def shuffle_respectively(X: pd.DataFrame, y: pd.DataFrame) -> tuple:
    """
    Function that shuffles X and y arrays respectively

    :param X: pandas dataframe containing input values
    :param y: pandas dataframe containing target variable
    :return: tuple containing shuffled X and y
    """
    df = pd.concat([X, y], axis = 1)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    return df.drop(columns=['target']), df[['target']]

def clean_data(X: pd.DataFrame, y: pd.DataFrame) -> tuple:
    """
    Function cleans the data, i. e. changes target to binary variables, drops redundant columns and cuts off beggining of true article

    :param X: pandas dataframe containing input values
    :param y: pandas dataframe containing target variable
    :return: tuple containing clean X and y arrays
    """
    y[y['target']=='Fake'] = 0
    y[y['target']=='True'] = 1
    y['target'] = y['target'].values.astype(int)
    
    X = X.drop(columns=['title', 'subject', 'date'])

    # Each article labeled as True is taken from Reuters and has 'city (Reuters) - ' on the beggining. We need to eliminate it!
    X.loc[y['target']==1, 'text'] = X.loc[y['target']==1, 'text'].apply(lambda x: ' '.join(x.split()[x.split().index('(Reuters)')+2:]) if '(Reuters)' in x.split() else x)
    return X, y