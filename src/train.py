import pickle
import os,sys
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from src.model import apply_encoder,split_data
from data.preprocess import load_data,save_preprocessed_df
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression



def train(x_train,y_train):
    # model=RandomForestClassifier(n_estimators=50, random_state=0)
    model=GaussianNB()
    # model = DecisionTreeClassifier(random_state=0)
    
    model = LogisticRegression()
    model.fit(x_train,y_train)
    return model

def save_model(model,path='model/model.pkl'):
    with open(path, 'wb') as file:
        pickle.dump(model, file)
    
    
def main():
    df= load_data('data/preprocessed1.csv')
    # print(df)
    print(df.columns)
    x_train, x_test, y_train, y_test=split_data(df)


    x_train,y_train,x_test,y_test=apply_encoder(x_train,x_test,y_train,y_test)
    model=train(x_train,y_train)
    save_model(model)

    predictions=model.predict(x_test)
    print(f"predictions:{predictions}")
    print()
    print(f"ytest:{y_test}")


main()