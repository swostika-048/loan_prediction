from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os
import sys
print("Current Working Directory:", os.getcwd())
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# from data.preprocess import load_data
from data.preprocess import load_data,save_preprocessed_df
label_encoder=LabelEncoder()

def get_xy(df):
    x= df.drop(columns=['Loan_Status'])
    y=df['Loan_Status']
    return x,y


def split_data(df, test_size=0.2, random_state=0):
    """
    Splits the DataFrame into training and testing sets.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame.
    target_column (str): The name of the target column.
    test_size (float): The proportion of the dataset to include in the test split.
    random_state (int): Controls the shuffling applied to the data before applying the split.
    
    Returns:
    x_train, x_test, y_train, y_test: Split data sets.
    """
    
    x,y= get_xy(df)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state)
    
    return x_train, x_test, y_train, y_test



def apply_encoder(x_train,x_test,y_train,y_test):
    df_columns=x_train.select_dtypes(include=['object']).columns
    # print()
    # print(f"testcol:{x_test.columns}")

    for col in  df_columns:
        # print(f"col:{col}")
        x_train[col]=label_encoder.fit_transform(x_train[col])
        
        x_test[col]=label_encoder.fit_transform(x_test[col])
    y_train=label_encoder.fit_transform(y_train)
    y_test=label_encoder.fit_transform(y_test)
    print(f":type:{type(y_test)}")
        # print(f"{col}:{df[col]}:")
    # a=y_test[0]
    # print(a)
    

    # print(x_train.shape[1])

    # save_preprocessed_df(df,path='data/encoded.csv')
    return x_train,y_train,x_test,y_test



# df=load_data('data/preprocessed.csv')

# x_train, x_test, y_train, y_test=split_data(df)
# # print(x_train)
# apply_encoder(x_train,x_test,y_train,y_test)
# # print(df.select_dtypes(include=['object']).columns)

