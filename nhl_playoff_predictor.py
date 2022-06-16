# All import statements
import configparser
import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split

def collect_data(folder):
    
    # Create blank dataframe to hold all data
    collected_data = pd.DataFrame()

    # Convert inputted folder string into a directory path
    directory = os.fsencode(folder)
    
    # Loop through all files in folder
    for file in os.listdir(directory):

        # Decode file and directory path from bytes to string and store the season year for future use
        file = file.decode('utf-8')
        folder = directory.decode('utf-8')
        year = file[:4]
        
        # Read the file into a dataframe
        df_current_season = pd.read_excel(folder + '/' + file)

        # Add the season year to the dataframe
        df_current_season['Year'] = year

        # Add dataframe into combined dataframe
        collected_data = pd.concat([collected_data, df_current_season])

    return collected_data

def clean_season_data(season_data):

    # Remove asterisks from Team column in season data, remove Rank column
    season_data.drop(columns = ['Rk'], inplace = True)
    season_data.rename(columns = {'Unnamed: 1': 'Team'}, inplace = True)
    season_data['Team'] = season_data['Team'].str.replace('*', '')

    # Drop columns that contain the same data for each team or are summarized by other columns
    season_data.drop(columns = ['GP', 'W', 'L', 'OL', 'PTS%', 'GF', 'GA', 'SOW', 'SOL'], inplace = True)

    return season_data

def clean_playoff_data(playoff_data):
    
    # Remove unnamed columns
    playoff_data = playoff_data[['Team 1', 'Team 2', 'Winner', 'Year']]

    # Trim leading/trailing spaces from columns
    playoff_data = playoff_data.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    
    return playoff_data

def combine_dataframes(season_data, playoff_data):

    # Merge Team 1 data and add Team 1 prefix for merged columns
    df_combined = pd.merge(playoff_data, season_data, how = 'inner', left_on = ['Team 1', 'Year'], right_on = ['Team', 'Year'])
    new_column_names = [(i, 'Team 1 ' + i) for i in df_combined.iloc[:, 4:].columns.values]
    df_combined.rename(columns = dict(new_column_names), inplace = True)

    # Merge Team 2 data and add Team 2 prefix for merged columns
    df_combined = pd.merge(df_combined, season_data, how = 'inner', left_on = ['Team 2', 'Year'], right_on = ['Team', 'Year'])
    new_column_names = [(i, 'Team 2 ' + i) for i in df_combined.iloc[:, 26:].columns.values]
    df_combined.rename(columns = dict(new_column_names), inplace = True)

    return df_combined

def create_ml_dataset(final_data):

    # Choose independent variables to be used for ML model
    X = final_data.iloc[:, np.r_[5:26, 27:48]]

    # Choose dependent variable the ML model will predict
    Y = final_data['Winner']
    
    # Divide model into training and testing splits
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state = 69)

    return X_train, X_test, Y_train, Y_test

# Set current working directory
cwd = os.getcwd()

# Read config file
config = configparser.ConfigParser()
config.read(cwd + '\\nhl_config.ini')

# Collect season stats and playoff results data
season_data = collect_data(config['DEFAULT']['SeasonStatsFolder'])
playoff_data = collect_data(config['DEFAULT']['PlayoffResultsFolder'])

# Clean season stats and playoff results data
cleaned_season_data = clean_season_data(season_data)
cleaned_playoff_data = clean_playoff_data(playoff_data)

# Create dataframe combining playoff results and season stats
final_data = combine_dataframes(season_data = cleaned_season_data, playoff_data = cleaned_playoff_data)

# Create dataset for machine learning
X_train, X_test, Y_train, Y_test = create_ml_dataset(final_data = final_data)