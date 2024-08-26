import pickle
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from src.model import apply_encoder,split_data
from data.preprocess import load_data, save_preprocessed_df



def evaluate_model(model, x_train, y_train, x_test, y_test):
    # Predictions on train and test sets
    train_predictions = model.predict(x_train)
    print(f"train_predictions:{train_predictions}")
    print(f"ytrain:{y_train}")
    test_predictions = model.predict(x_test)

    
    # Calculate training and validation accuracy
    train_accuracy = accuracy_score(y_train, train_predictions)
    test_accuracy = accuracy_score(y_test, test_predictions)
    
    # Generate classification reports
    train_report = classification_report(y_train, train_predictions, target_names=['Not Approved', 'Approved'])
    test_report = classification_report(y_test, test_predictions, target_names=['Not Approved', 'Approved'])
    
    # Generate confusion matrices
    train_conf_matrix = confusion_matrix(y_train, train_predictions)
    test_conf_matrix = confusion_matrix(y_test, test_predictions)
    
    return train_accuracy, test_accuracy, train_report, test_report, train_conf_matrix, test_conf_matrix

# Load data
df = load_data('data/preprocessed.csv')

# Split data into train and test sets
x_train, x_test, y_train, y_test=split_data(df)

# Apply encoding
x_train,y_train,x_test,y_test = apply_encoder(x_train, x_test, y_train, y_test)

# # Train model
# model = train(x_train, y_train)


model_path='model/model.pkl'
with open(model_path, 'rb') as file:
    model = pickle.load(file)
# Evaluate model
train_accuracy, test_accuracy, train_report, test_report, train_conf_matrix, test_conf_matrix = evaluate_model(model, x_train, y_train, x_test, y_test)

# Print evaluation results
print(f"Training Accuracy: {train_accuracy:.4f}")
print(f"Validation Accuracy: {test_accuracy:.4f}")
print("\nTraining Classification Report:")
print(train_report)
print("Training Confusion Matrix:")
print(train_conf_matrix)

print("\nValidation Classification Report:")
print(test_report)
print("Validation Confusion Matrix:")
print(test_conf_matrix)
