import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
import os
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(file_path='data/dataset/data.xlsx'):
    """
    Load the loan eligibility data from a CSV file.

    Parameters:
    file_path (str): Path to the CSV file containing the data.

    Returns:
    DataFrame: Loaded data as a pandas DataFrame.

    """
    file_extension = os.path.splitext(file_path)[1]
    if file_extension == '.xlsx' or file_extension == '.xls':
        df = pd.read_excel(file_path)
    elif file_extension == '.csv':
        df = pd.read_csv(file_path)
    else:
        raise ValueError("Unsupported file format: {}".format(file_extension))
    # print(df)

    return df


def visualize_dataset(df):
    # Display basic information
    # print("First few rows of the dataset:")
    # print(df.head())
    
    print("\nSummary statistics:")
    print(df.describe())
    
    # Visualize distributions of numerical features
    num_features = df.select_dtypes(include=['float64', 'int64']).columns
    print(f"num_features:{num_features}")
    for feature in num_features:
        plt.figure(figsize=(10, 4))
        sns.histplot(df[feature], kde=True, bins=40)
        plt.title(f'Distribution of {feature}')
        plt.xlabel(feature)
        plt.ylabel('Frequency')
        # plt.show()
        plt.savefig(f"data/plot/{feature}.png")
    
    # Visualize distributions of categorical features
    cat_features = df.select_dtypes(include=['object']).columns
    for feature in cat_features:
        plt.figure(figsize=(10, 4))
        sns.countplot(y=feature, data=df)
        plt.title(f'Distribution of {feature}')
        plt.xlabel('Count')
        plt.ylabel(feature)
        # plt.show()
        plt.savefig(f"data/plot/{feature}.png")
    

def check_null_and_fill(df):
   null_info= df.isnull().sum()
   print(f"null_info{null_info}")
   num_columns=df.columns
   for feature in num_columns:
       if df[feature].isnull().sum()>0:
           print(f"{df[feature].head()}")
           print(f"filling the null value in{feature}")
           df[feature].fillna(df[feature].mode()[0],inplace=True)
   null_info= df.isnull().sum()
   visualize_dataset(df)
   print(f"null_info{null_info}")
   return df

def drop_column(df, drop_columns=None):
    """
    Loads the dataset from a CSV file and drops specified columns.
    
    Parameters:
    file_path (str): The path to the CSV file.
    drop_columns (list): List of column names to drop from the DataFrame.
    
    Returns:
    df (pd.DataFrame): The preprocessed DataFrame.
    """
    drop_columns='Loan_ID'
    if drop_columns:
        df.drop(columns=drop_columns, inplace=True)
    
    return df

def save_preprocessed_df(df,path='data/preprocessed1.csv'):
    df.to_csv(path,index=False)

def main():
    df=load_data()
    print(df)
    df= drop_column(df)
    df=check_null_and_fill(df)
    save_preprocessed_df(df)

main()