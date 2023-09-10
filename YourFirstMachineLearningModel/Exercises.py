import pandas as pd

# Path of the file to read
iowa_file_path = '../input/home-data-for-ml-course/train.csv'

home_data = pd.read_csv(iowa_file_path)

# Set up code checking
from learntools.core import binder
binder.bind(globals())
from learntools.machine_learning.ex3 import *

print("Setup Complete")



# 1. Specify Prediction Target Select the target variable, which corresponds to the sales price. Save this to a new variable called y. You'll need to print a list of the columns to find the name of the column you need.
# print the list of columns in the dataset to find the name of the prediction target
print(home_data.columns)
y = home_data.SalePrice
# Check your answer
step_1.check()

# 2. Create X Now you will create a DataFrame called X holding the predictive features. Since you want only some columns from the original data, you'll first create a list with the names of the columns you want in X. You'll use just the following columns in the list (you can copy and paste the whole list to save some typing, though you'll still need to add quotes):
# Create the list of features below
feature_names = ["LotArea", "YearBuilt", "1stFlrSF", "2ndFlrSF", "FullBath", "BedroomAbvGr", "TotRmsAbvGrd"]
# Select data corresponding to features in feature_names
X = home_data[feature_names]
# Check your answer
step_2.check()

# 3. Review Data Before building a model, take a quick look at X to verify it looks sensible
# print description or statistics from X
print(X.describe())
# print the top few lines
print(X.head())

# 4. Specify and Fit Model Create a DecisionTreeRegressor and save it iowa_model. Ensure you've done the relevant import from sklearn to run this command. Then fit the model you just created using the data in X and y that you saved above.
from sklearn.tree import DecisionTreeRegressor
#specify the model. 
#For model reproducibility, set a numeric value for random_state when specifying the model
iowa_model = DecisionTreeRegressor(random_state=1)
# Fit the model
print(iowa_model.fit(X, y))
# Check your answer
step_3.check()

# 5. Make Predictions Make predictions with the model's predict command using X as the data. Save the results to a variable called predictions.
predictions = iowa_model.predict(X)
print(predictions)
# Check your answer
step_4.check()
