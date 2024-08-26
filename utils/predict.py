import pickle

# Load the model from the file
with open('model/model.pkl', 'rb') as file:
    model = pickle.load(file)