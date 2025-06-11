import pandas as pd
import os
from sklearn.model_selection import train_test_split


def load_data(name: str) -> pd.DataFrame:
    """
    Load dataset from a CSV file and add a 'target' column from the filename.

    Args:
        name (str): Filename of the CSV file with extension.

    Returns:
        pd.DataFrame: Dataframe where each row represents a single article with
        attributes: title, text, subject, date and added 'target' column.
    """
    class_name = name.removesuffix(".csv")
    data = pd.read_csv(os.path.join("Datasets", name))
    data['target'] = class_name
    return data


def choose_random_part(data: pd.DataFrame, n: int) -> pd.DataFrame:
    """
    Randomly select a subset of articles from the dataset.

    Args:
        data (pd.DataFrame): Dataframe containing the full dataset.
        n (int): Number of articles to randomly sample.

    Returns:
        pd.DataFrame: DataFrame containing the sampled articles.
    """
    return data.sample(n, random_state=42)


def divide_data_into_train_and_test(
        data: pd.DataFrame,
        t_size: float
        ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split dataset into training and testing datasets.

    Args:
        data (pd.DataFrame): Dataframe containing the full dataset, including the 'target' column.
        t_size (float): Proportion of the dataset to include in test split (between 0.0 and 1.0).

    Returns:
        tuple: A tuple containing:
            - X_train (pd.DataFrame): Training input features.
            - X_test (pd.DataFrame): Testing input features.
            - y_train (pd.DataFrame): Training target values.
            - y_test (pd.DataFrame): Testing target values.
    """
    X = data.drop(columns=["target"])
    y = data[["target"]]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=t_size, random_state=42
    )
    return X_train, X_test, y_train, y_test


def merge_datasets(dataset1: pd.DataFrame, dataset2: pd.DataFrame) -> pd.DataFrame:
    """
    Merge two datasets into a single pandas DataFrame.

    Args:
        dataset1 (pd.DataFrame): The first dataset.
        dataset2 (pd.DataFrame): The second dataset.

    Returns:
        pd.DataFrame: A merged DataFrame with reset index.
    """
    return pd.concat([dataset1, dataset2], ignore_index=True)


def shuffle_respectively(X: pd.DataFrame, y: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Shuffle X and y dataframes in unison while preserving their row alignment.

    Args:
        X (pd.DataFrame): Dataframe containing input features.
        y (pd.DataFrame): Dataframe containing target labels.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: Tuple of shuffled
        X and y DataFrames.
    """
    df = pd.concat([X, y], axis=1)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    return df.drop(columns=['target']), df[['target']]


def clean_data(X: pd.DataFrame, y: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Clean the input features and target labels.

    This function:
    - Converts the target labels to binary (0 for 'Fake', 1 for 'True')
    - Drops unnecessary columns: 'title', 'subject', and 'date'
    - Removes introductory text like 'City (Reuters) - ' from the beginning of true news articles

    Args:
        X (pd.DataFrame): Dataframe containing the input features.
        y (pd.DataFrame): Dataframe containing the target labels.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: Tuple containing cleaned feature and label DataFrames.
    """
    y['target'] = y['target'].map({'Fake': 0, 'True': 1}).astype(int)
    X = X.drop(columns=['title', 'subject', 'date'])

    # Each article labeled as True is taken from Reuters and has 'city (Reuters) - '
    # on the beggining. We need to eliminate it!
    X.loc[y['target'] == 1, 'text'] = X.loc[y['target'] == 1, 'text'].apply(
        lambda x: ' '.join(x.split()[x.split().index('(Reuters)')+2:]) if '(Reuters)' in x.split() else x
        )
    return X, y
