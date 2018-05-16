#
# 	Project: Project Pseudo Code
#
#	Program: accident_analysis_model
# 	Purpose: Learning Bayesian network for dynamic driver risk assessment
# 	Description:
#  		This program uses multiple APIs to make Bayesian networks and
#		generate predictions to test the models' accuracies. The first
#		model is made using pgmpy Python library. The model is given
#		every node and edge in the graph. The network is then fit to
#		training data to generate each variables conditional
#		probability distribution. Inference is then done using the
#		variable elimination algorithm and validation data is compared
#		against predictions made. The second model is made using the
#		pomegranate Python library. The library is able to generate a
#		Bayesian network structure conditional probability distributions
#		using a learning score for each possible structure. The API also
#		has a predict function used to compare the generated model to
#		the validation dataset. The structure and prediction accuracy of
#		these models will then be compared.
#
# 	Programmer: Nathan Turner
# 	Organization: JHU Part-Time Programs in Engineering and Applied Science
# 	Language: Python
# 	Date Created: 04/03/18

# PGM libraries
from pgmpy.models import BayesianModel
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination

# utility functions
from utilities import *

import pandas as pd
import argparse
import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split


##############################################################################################
# Get Command Line Arguments
##############################################################################################
parser = argparse.ArgumentParser(description='csv to postgres', fromfile_prefix_chars="@")
parser.add_argument('--accident_file', dest='accident_csv_file', help='accident csv file to import', action='store',
                    default='data/dftRoadSafety_Accidents_2016.csv')
parser.add_argument('--driver_file', dest='driver_vehicle_csv_file',
                    help='driver and vehicle information csv file to import', action='store',
                    default='data/driver_vehicle_data.csv')
parser.add_argument('--checkpoint_dir', dest='checkpoint_dir',
                    help='directory to save data, model and prediction checkpoints', action='store',
                    default='checkpoints/')
parser.add_argument('--no_dataset_plots', dest='plot_dataset',
                    help='flag to plot distributions and densities of incoming data', action='store_false',
                    default=True)
parser.add_argument('--test_program', dest='test_program',
                    help='flag greatly reduce the ammount of records used to make and predict on, will shorten runtime dramatically',
                    action='store_true',
                    default=False)
args = parser.parse_args()


##############################################################################################
# Get Dataset
##############################################################################################
try:
    dataset = pd.read_pickle(args.checkpoint_dir + 'data.pkl')
    print 'Loaded dataset of size ' + str(dataset.shape[0]) + '...'
except:
    # if dataset is not saved, we need to import csv files and preprocess
    print 'Reading and processing dataset...'
    # import data, if test_program is set to true, a fraction of the data will be imported to greatly reduce runtime
    dataset = import_data(args.accident_csv_file, args.driver_vehicle_csv_file, test_program=args.test_program)
    # save dataset so we do not have to wait to process entire set every run
    dataset.to_pickle(args.checkpoint_dir + 'data.pkl')

# plot visualizations of incoming data
if args.plot_dataset:
    plot_data_distributions(dataset)

# print correlations of all variables to accident severity
# Find correlations and sort
accident_node = 'Accident_Severity'
correlation = dataset.corr()[accident_node].sort_values()
print '\nCorrelation:\n' + str(correlation)

# drop factors that are not correlated to accident severity, done to speed up processing time
dataset, possible_values = reduce_dataset(dataset)


##############################################################################################
# Make Bayesian Model
##############################################################################################

# Declare edges of network
assumed_graph_edges = [
    ('Light_Conditions', 'Accident_Severity'),
    ('Road_Type', 'Speed_limit'),
    ('Speed_limit', 'Accident_Severity'),
    ('Urban_or_Rural_Area', 'Accident_Severity'),
    ('Sex_of_Driver', 'Accident_Severity'),
    ('Age_of_Driver', 'Accident_Severity'),
    ('Age_of_Vehicle', 'Accident_Severity'),
    ('Driver_Home_Area_Type', 'Accident_Severity')
]


# Partition data into training and validation data, using 25% for validation
train, test = train_test_split(dataset, test_size=0.25)
print '\nTraining dataset summary:'
print train.describe()

# try to load model
my_file = Path(args.checkpoint_dir + 'model.pkl')
if my_file.exists():
    with open(args.checkpoint_dir + 'model.pkl', 'rb') as input:
        model = pickle.load(input)
else:
    # make bayesian model
    model = BayesianModel(assumed_graph_edges)
    print('Fitting...')
    # get model cpds from train data
    model.fit(train, estimator=MaximumLikelihoodEstimator, state_names=possible_values)
    print('Done Fitting...')

    # save model so we don't have to make it again
    with open(args.checkpoint_dir + 'model.pkl', 'wb') as output:
        pickle.dump(model, output, pickle.HIGHEST_PROTOCOL)

print '\nConditional probability distributions of model:'
print model.get_cpds()


##############################################################################################
# Make Predictions from Bayesian Model
##############################################################################################

# get truth data and drop from test set, needed to predict missing data
y_test = test['Accident_Severity']
predict_data = test.drop(columns='Accident_Severity')

# try to load predictions
my_file = Path(args.checkpoint_dir + 'predictions.pkl')
if my_file.exists():
    with open(args.checkpoint_dir + 'predictions.pkl', 'rb') as input:
        predict = pickle.load(input)
else:
    # perform variable elimination
    print('Performing Variable Elimination...')
    model_inference = VariableElimination(model, state_names=possible_values)

    # predict using model
    print('Predicting...')
    predict = model.predict(predict_data)

    # save predictions so we don't have to make them again
    with open(args.checkpoint_dir + 'predictions.pkl', 'wb') as output:
        pickle.dump(predict, output, pickle.HIGHEST_PROTOCOL)

# evaluate predictions
evaluate(predict, y_test)