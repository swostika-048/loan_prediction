import pickle
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from src.model import apply_encoder,split_data
from data.preprocess import load_data, save_preprocessed_df


label_mapping = {
    0: 'Not Approved',
    1: 'Approved'
}

def prediction(model,data):
    # train_predictions=model.predict(x_train)
    # test_predictions=model.predict(x_test)
    prediction=model.predict(data)
    print(f"completed prediction")
    return prediction

def evaluate_model(model, x_train, y_train, x_test, y_test):
    # Predictions on train and test sets
    train_predictions=prediction(model,x_train)
    test_predictions=prediction(model,x_test)
    # train_predictions = model.predict(x_train)
    print(f"train_predictions:{train_predictions}")
    print(f"ytest:{y_train}")
    # test_predictions = model.predict(x_test)
    categorical_predictions = [label_mapping[pred] for pred in test_predictions]
    print(categorical_predictions)

    
    # Calculate training and validation accuracy
    train_accuracy = accuracy_score(y_train, train_predictions)
    train_loss=1-train_accuracy
    test_accuracy = accuracy_score(y_test, test_predictions)
    test_loss=1-test_accuracy
    
    # Generate classification reports
    train_report = classification_report(y_train, train_predictions, target_names=['Not Approved', 'Approved'])
    test_report = classification_report(y_test, test_predictions, target_names=['Not Approved', 'Approved'])
    
    # Generate confusion matrices
    train_conf_matrix = confusion_matrix(y_train, train_predictions)
   
    test_conf_matrix = confusion_matrix(y_test, test_predictions)
    plot_confusion_matrix(y_test,test_predictions,list(label_mapping))

    
    return train_accuracy, test_accuracy, train_report, test_report, train_conf_matrix, test_conf_matrix,train_loss,test_loss

def plot_confusion_matrix(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()


def main():
    # Load data
    df = load_data('data/preprocessed1.csv')

    # Split data into train and test sets
    x_train, x_test, y_train, y_test=split_data(df)

    # Apply encoding
    x_train,y_train,x_test,y_test = apply_encoder(x_train, x_test, y_train, y_test)



    model_path='model/dtc.pkl'
    with open(model_path, 'rb') as file:
        model = pickle.load(file)


    # Evaluate model
    train_accuracy, test_accuracy, train_report, test_report, train_conf_matrix, test_conf_matrix,train_loss,test_loss = evaluate_model(model, x_train, y_train, x_test, y_test)

    # Print evaluation results
    print(f"Training Accuracy: {train_accuracy:.4f}")
    print(f"training loss: {train_loss:.4f}")
    print(f"Validation Accuracy: {test_accuracy:.4f}")
    print(f"Validation loss: {test_loss:.4f}")
    print("\nTraining Classification Report:")
    print(train_report)
    print("Training Confusion Matrix:")
    print(train_conf_matrix)

    print("\nValidation Classification Report:")
    print(test_report)
    print("Validation Confusion Matrix:")
    print(test_conf_matrix)


# main()