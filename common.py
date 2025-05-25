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
    y = data["target"]
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

def preprocess_data (data):
    # TODO
    return