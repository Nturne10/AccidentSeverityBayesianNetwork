import matplotlib.pyplot as plt
import seaborn as sns


def plot_data_distributions(dataset):
    """
    This function plots various distributions and densities to analyze the structure of the incoming data.
    """

    accident_node = 'Accident_Severity'
    
    # histogram of accident severity
    plt.hist(dataset[accident_node], bins=3)
    plt.xlabel('Accident Severity')
    plt.ylabel('Count')
    plt.title('Distribution of Accident Severity')
    plt.show()

    # histogram of age
    plt.hist(dataset['Age_of_Driver'], bins=20)
    plt.xlabel('Age_of_Driver')
    plt.ylabel('Count')
    plt.title('Distribution of Driver Age')
    plt.show()

    # histogram of day of week
    plt.hist(dataset['Day_of_Week'], bins=20)
    plt.xlabel('Day_of_Week')
    plt.ylabel('Count')
    plt.title('Distribution of Day of Week')
    plt.show()

    # Make one plot for each different location
    sns.kdeplot(dataset.ix[dataset['Urban_or_Rural_Area'] == 1, accident_node],
                label = 'Urban', shade = True)
    sns.kdeplot(dataset.ix[dataset['Urban_or_Rural_Area'] == 2, accident_node],
                label = 'Rural', shade = True)
    # Add labeling
    plt.xlabel('Accident Severity')
    plt.ylabel('Density')
    plt.title('Density Plot of Accident Severity by Location')
    plt.show()

    # Make one plot for each different location
    sns.kdeplot(dataset.ix[dataset['Age_of_Driver'] < 30, accident_node],
                label = '< 30', shade = True)
    range_df = dataset['Age_of_Driver'].between(30,50,inclusive=True)
    sns.kdeplot(dataset.ix[range_df, accident_node], label = '30<= age <=50', shade = True)
    range_df = dataset['Age_of_Driver'].between(50,65,inclusive=True)
    sns.kdeplot(dataset.ix[range_df, accident_node], label = '50<= age <=65', shade = True)
    sns.kdeplot(dataset.ix[dataset['Age_of_Driver'] > 65, accident_node],
                label = '> 65', shade = True)
    # Add labeling
    plt.xlabel('Accident Severity')
    plt.ylabel('Density')
    plt.title('Density Plot of Accident Severity by Age Group')
    plt.show()