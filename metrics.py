from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# Pull the test data into dataframe (imagine this was a database query)
y_test = get_data(test=True)[1]

# Load the predictions
# TODO: Load the output from predict.py

# Calculate model prediction metrics
r2_score(y_test, predicted_qualities)
mean_squared_error(y_test, predicted_qualities)
mean_absolute_error(y_test, predicted_qualities)

# Output
# TODO: Store these metrics and use them to evaluate the model
