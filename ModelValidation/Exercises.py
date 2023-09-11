import pandas as pd
from sklearn.tree import DecisionTreeRegressor

# Path of the file to read
iowa_file_path = '../input/home-data-for-ml-course/train.csv'

home_data = pd.read_csv(iowa_file_path)
y = home_data.SalePrice
feature_columns = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
X = home_data[feature_columns]

# Specify Model
iowa_model = DecisionTreeRegressor()
# Fit Model
iowa_model.fit(X, y)

print("First in-sample predictions:", iowa_model.predict(X.head()))
print("Actual target values for those homes:", y.head().tolist())

# Set up code checking
from learntools.core import binder
binder.bind(globals())
from learntools.machine_learning.ex4 import *
print("Setup Complete")


# 1. Split Your Data Use the train_test_split function to split up your data. Give it the argument random_state=1 so the check functions know what to expect when verifying your code. Recall, your features are loaded in the DataFrame X and your target is loaded in y.
# Import the train_test_split function and uncomment
from sklearn.model_selection import train_test_split

# fill in and uncomment
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 1)
# Check your answer
step_1.check()

# 2. Specify and Fit the Model Create a DecisionTreeRegressor model and fit it to the relevant data. Set random_state to 1 again when creating the model.
# You imported DecisionTreeRegressor in your last exercise
# and that code has been copied to the setup code above. So, no need to
# import it again
# Specify the model
iowa_model = DecisionTreeRegressor(random_state = 1)
# Fit iowa_model with the training data.
iowa_model.fit(train_X, train_y)
# Check your answer
step_2.check()

# 3. Make Predictions with Validation data
# Predict with all validation observations
val_predictions = iowa_model.predict(val_X)
# Check your answer
step_3.check()
# Inspect your predictions and actual values from validation data.
# print the top few validation predictions
print(iowa_model.predict(X.head()))
# print the top few actual prices from validation data
print(iowa_model.predict(val_X.head()))

# 4. Calculate the Mean Absolute Error in Validation Data
from sklearn.metrics import mean_absolute_error
val_mae = mean_absolute_error(val_y, val_predictions)
# uncomment following line to see the validation_mae
print(val_mae)
# Check your answer
step_4.check()
