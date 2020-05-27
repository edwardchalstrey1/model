import numpy as np
from sklearn.linear_model import BayesianRidge
import pickle
from get_data import get_data


# Pull the training data into dataframes (imagine this was a database query)
X_train, y_train = get_data()

# Set model parameters
alpha_1=1e-06
alpha_2=1e-06
n_iter=500

# Create and train model
model = BayesianRidge(alpha_1=alpha_1, alpha_2=alpha_2, n_iter=n_iter)
model.fit(X_train, y_train)

# Save model to disk
# TODO: could we save this to a database? - or use mlflow
filename = 'finalized_model.sav'
pickle.dump(model, open(filename, 'wb'))

# Save the parameters
# TODO: Save these somehow (in a file, db, with mlflow?)
