import pandas as pd
import argparse
import pickle
import numpy as np


# specify possible values of each factor (needed for prediction)
data = {
    'Light_Conditions': np.array([0]),
    'Urban_or_Rural_Area': np.array([1]),
    'Accident_Severity': np.array([0]),
    'Speed_limit': np.array([2]),
    'Road_Type': np.array([0]),
    'Age_of_Driver': np.array([3]),
    'Age_of_Vehicle': np.array([0]),
    'Sex_of_Driver': np.array([0]),
    'Driver_Home_Area_Type': np.array([0])
}


##############################################################################################
# Get Command Line Arguments
##############################################################################################
parser = argparse.ArgumentParser(description='csv to postgres', fromfile_prefix_chars="@")
parser.add_argument('--checkpoint_dir', dest='checkpoint_dir',
                    help='directory to save data, model and prediction checkpoints', action='store',
                    default='checkpoints/')
args = parser.parse_args()


##############################################################################################
# Make Bayesian Model
##############################################################################################

# try to load model
try:
    with open(args.checkpoint_dir + 'model.pkl', 'rb') as input:
        model = pickle.load(input)
except:
    raise ValueError('No model made yet')


##############################################################################################
# Make Predictions from Bayesian Model
##############################################################################################
young_driver_test = pd.DataFrame(data)


# get truth data and drop from test set, needed to predict missing data
y_test = young_driver_test['Accident_Severity']
predict_data = young_driver_test.drop(columns='Accident_Severity')

# predict using model
print('Predicting...')
predict = model.predict(predict_data)

print 'Using the following data as input:'
print data
print 'Predicting accident severity of:'
print predict['Accident_Severity'][0]