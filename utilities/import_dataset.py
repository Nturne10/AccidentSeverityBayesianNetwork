import pandas as pd
import numpy as np
import csv


def import_data(accident_csv_file, driver_vehicle_csv_file, test_program=False):
    """
    This function reads in two csv files containing records of car accidents in the UK for 2016. Missing
    data is removed from the set. Factors that have a large range are binned into ranges to make the
    prediction step faster. A pandas dataframe is returned with all of the data inside.
    """

    # include headers from conditions csv
    conditions_included_cols = [
        'Location_Easting_OSGR',
        'Location_Northing_OSGR',
        'Accident_Severity',
        'Day_of_Week',
        'Time',
        'Road_Type',
        'Speed_limit',
        'Light_Conditions',
        'Weather_Conditions',
        'Road_Surface_Conditions',
        'Special_Conditions_at_Site',
        'Urban_or_Rural_Area',
    ]
    # include headers from vehicle/driver csv
    veh_driver_included_cols = [
        'Vehicle_Type',
        'Sex_of_Driver',
        'Age_of_Driver',
        'Age_of_Vehicle',
        'Driver_IMD_Decile',  # index of multiple deprivation to define how deprived driver is
        'Driver_Home_Area_Type',
        'Vehicle_IMD_Decile'
    ]

    # id of row to match to other data
    id_col_name = 'Accident_Index'
    time_name = 'Time'
    vehicle_ref_name = 'Vehicle_Reference'  # 1 if primary vehicle in accident
    speed_limit_name = 'Speed_limit'
    driver_age_name = 'Age_of_Driver'
    vehicle_age_name = 'Age_of_Vehicle'
    # open conditions csv file
    raw_conditions_data = []
    with open(accident_csv_file, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            raw_conditions_data.append(row)

    # get header row of csv and delete the row from the raw data
    header = raw_conditions_data[0]
    del raw_conditions_data[0]

    # get index of all columns to use for data
    conditions_cols_index_list = [header.index(element) for element in conditions_included_cols]
    # id col will be used to match data from different csv sheets
    id_col = header.index(id_col_name)
    # only using hour integer of time (special case)
    time_indx = header.index(time_name)
    # split speed limit into 3 categories for fitting model
    speed_index = header.index(speed_limit_name)


    # keep track of how many rows have missing data
    total_skipped_rows = 0

    # loop over entire dataset and get necessary cols
    accident_id_content = {}
    for row in raw_conditions_data:
        col_data = []
        skip_row = False
        # loop over index of every col needed for dataset
        for col in conditions_cols_index_list:
            element = row[col]
            # missing elements are named -1, NULL or are empty
            if element == '-1' or element == 'NULL' or element == '':
                total_skipped_rows += 1
                skip_row = True
                break
            # only use hour of time to keep every value an integer
            if col == time_indx:
                hours, minutes = element.split(':')
                element = hours

            # make sure to save values as integers
            element = int(element)

            # split speed limit into 3 categories for fitting model
            if col == speed_index:
                if element < 30:
                    element = 0
                elif element < 55:
                    element = 1
                else:
                    element = 2

            col_data.append(element)

        # add data to dataset if not missing any values
        if not skip_row:
            accident_id_content[row[id_col]] = col_data

    # print dataset info
    print 'Read Conditions Dataset:'
    print 'Data Size: ' + str(len(accident_id_content))
    print 'Dropped rows from missing data: ' + str(total_skipped_rows)


    # open vehicle/driver csv file
    raw_driver_data = []
    with open(driver_vehicle_csv_file, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            raw_driver_data.append(row)

    # get header row of csv and delete the row from the raw data
    header = raw_driver_data[0]
    del raw_driver_data[0]

    # get index of all columns to use for data
    driver_cols_index_list = [header.index(element) for element in veh_driver_included_cols]
    # id col will be used to match data from different csv sheets
    id_col = header.index(id_col_name)
    # only use one vehicle in report (1)
    vehicle_ref_index = header.index(vehicle_ref_name)
    # split driver age into 4 categories
    driver_age_index = header.index(driver_age_name)
    # split vehicle age into 3 categories
    vehicle_age_index = header.index(vehicle_age_name)

    # keep track of how many rows don't have matching accident ids to combine csvs
    total_unmatched_rows = 0

    # loop over entire dataset and get necessary cols
    driver_id_content = {}
    for row in raw_driver_data:
        col_data = []
        skip_row = False
        # loop over index of every col needed for dataset
        for col in driver_cols_index_list:
            # skip row if not primary vehicle
            if col == vehicle_ref_index and row[col] != '1':
                skip_row = True
                break
            element = row[col]
            # missing elements are named -1, NULL or are empty
            if element == '-1' or element == 'NULL' or element == '':
                total_skipped_rows += 1
                skip_row = True
                break

            # make sure to save values as integers
            element = int(element)

            # split driver age into 4 categories
            if col == driver_age_index:
                if element < 25:
                    element = 0
                elif element < 40:
                    element = 1
                elif element < 60:
                    element = 2
                else:
                    element = 3

            # split vehicle age into 3 categories
            if col == vehicle_age_index:
                if element < 5:
                    element = 0
                elif element < 10:
                    element = 1
                else:
                    element = 2

            col_data.append(element)

        # add data to dataset if not missing any values
        if not skip_row:
            driver_id_content[row[id_col]] = col_data

    dict_keys = list(accident_id_content.keys())
    for key in dict_keys:
        if key not in driver_id_content:
            total_unmatched_rows += 1
            accident_id_content.pop(key)
        else:
            accident_id_content[key].extend(driver_id_content[key])

    # print dataset info
    print '\nRead Driver Dataset...'
    print 'New Data Size: ' + str(len(accident_id_content))
    print 'Dropped rows from missing data: ' + str(total_skipped_rows)
    print 'Rows that do not match conditions data: ' + str(total_unmatched_rows)

    # make pandas dataframe from data, specify categories and values
    all_rows = list(accident_id_content.values())
    if test_program:
        all_rows = all_rows[:100]
    categories = conditions_included_cols + veh_driver_included_cols
    input_values = np.array([all_rows[row] for row in range(len(all_rows))])

    dataset = pd.DataFrame(input_values, columns=categories)
    return dataset