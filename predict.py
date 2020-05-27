import pickle


# Load the model from disk
model = pickle.load(open(filename, 'rb'))

# Pull the test data into dataframe (imagine this was a database query)
X_test = get_data(test=True)[0]

# Predict with model
predicted_qualities = model.predict(X_test)

# Save the predictions (perhaps this should be saved to a db or file?)
# predicted_qualities
