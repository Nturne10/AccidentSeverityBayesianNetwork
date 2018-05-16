import numpy as np
from sklearn.utils import shuffle


def reduce_dataset(dataset):
    """
    This function reduces the number of variables in the dataset. Only the strongly correlated nodes are kept.
    This is done to speed up the processing time involved with predicting data. Also, variables that are not
    strongly correlated to the accident severity could skew the predictions. This function also defines all
    possible values for a node in the graph to take.
    """

    # drop weakly correlated factors
    cols_to_drop = [
        'Road_Surface_Conditions',
        'Location_Easting_OSGR',
        'Location_Northing_OSGR',
        'Day_of_Week',
        'Weather_Conditions',
        'Special_Conditions_at_Site',
        'Vehicle_Type',
        'Time',
        'Driver_IMD_Decile',
        'Vehicle_IMD_Decile'
    ]

    # specify possible values of each factor (needed for prediction)
    state_names = {
        'Light_Conditions': [1, 4, 5, 6, 7],
        'Urban_or_Rural_Area': list(range(1, 4)),
        'Accident_Severity': list(range(1, 4)),
        'Speed_limit': list(range(0, 3)),
        'Road_Type': [1, 2, 3, 6, 7, 9, 12],
        'Age_of_Driver': list(range(0, 4)),
        'Age_of_Vehicle': list(range(0, 3)),
        'Sex_of_Driver': list(range(1, 4)),
        'Driver_Home_Area_Type': list(range(1, 4))
    }

    # specify the new possible values of each factor
    # make index start at zero
    new_state_names = {
        'Light_Conditions': np.array([0,1,2,3,4], dtype=np.int32),
        'Urban_or_Rural_Area': np.array([0,1,2], dtype=np.int32),
        'Accident_Severity': np.array([0,1,2], dtype=np.int32),
        'Speed_limit': np.array([0,1,2], dtype=np.int32),
        'Road_Type': np.array([0,1,2,3,4,5,6], dtype=np.int32),
        'Age_of_Driver': np.array([0,1,2,3], dtype=np.int32),
        'Age_of_Vehicle': np.array([0,1,2], dtype=np.int32),
        'Sex_of_Driver': np.array([0,1,2], dtype=np.int32),
        'Driver_Home_Area_Type': np.array([0,1,2], dtype=np.int32)
    }

    # make all values start at zero
    for key in state_names.keys():
        dataset[key] = dataset[key].replace(state_names[key], new_state_names[key])

    # drop columns from dataset
    dataset = dataset.drop(cols_to_drop, axis=1)
    # return shuffled dataset
    return shuffle(dataset), new_state_names
