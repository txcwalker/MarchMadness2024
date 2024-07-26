import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

#Importing the data, https://www.kaggle.com/datasets/andrewsundberg/college-basketball-dataset?select=cbb13.csv
madness_df = pd.read_csv('data/tournament_teams_2013_2023.csv')
#2024 Teams
madness_2024_df = pd.read_csv('data/cbb24.csv')
#Renaming Columns
madness_df = madness_df.rename(columns = lambda name: name.title())
madness_2024_df = madness_2024_df.rename(columns = lambda name: name.title())

#Import the teams
#There is no Louisana Lafayette in the file...
cbb_teams_df = pd.read_csv('data/MTeams.csv')
#Renaming Column
cbb_teams_df = cbb_teams_df.rename(columns = {'TeamName':'Team'})

#Define diciontaries for conference and Tournament round reached
#Conference
conferences_dict = {
    "ACC": 1,
    "AE": 31,
    "A10": 2,
    "Amer": 3,
    "ASun": 4,
    "BE": 5,
    "B10": 6,
    "B12": 7,
    "BSky": 8,
    "BSth": 9,
    "BW": 10,
    "CAA": 11,
    "CUSA": 12,
    "Horz": 13,
    "Ivy": 14,
    "MAAC": 15,
    "MAC": 16,
    "MEAC": 17,
    "MVC": 18,
    "MWC": 19,
    "NEC": 20,
    "OVC": 21,
    "P12": 22,
    "Pat": 23,
    "SB": 24,
    "SC": 25,
    "SEC": 26,
    "Slnd": 27,
    "Sum": 28,
    "SWAC": 29,
    "WAC": 30,
    "WCC": 33
}

#Apply the dictionary
madness_df['Conf'] = madness_df['Conf'].map(conferences_dict)
madness_2024_df['Conf'] = madness_2024_df['Conf'].map(conferences_dict)
#Tournament Round
postseason_dict = {
    "Champions": 1,
    "2ND": 2,
    "F4": 3,
    "E8": 4,
    "S16": 5,
    "R32": 6,
    "R64": 7
}

#Apply the dictionary
madness_df['Postseason'] = madness_df['Postseason'].map(postseason_dict)

#Merging to add in team IDs
madness_df = madness_df.merge(cbb_teams_df, how ='inner', on = 'Team')
madness_2024_df = madness_2024_df.merge(cbb_teams_df, how = 'inner', on = 'Team')

#Creating Team Name Dictionary
team_id_dict = dict(zip(madness_df['Team'], madness_df['TeamID']))

#Dropping Team Name
analysis_madness_df = madness_df.drop('Team', axis = 1)
analysis_2024_madness_df = madness_2024_df.drop('Team', axis = 1)

#Creating the Model
# Drop any rows with missing values
analysis_madness_df.dropna(inplace=True)

# Define features (X) and target variable (y)
X = analysis_madness_df.drop('Postseason', axis=1)  # Features
y = analysis_madness_df['Postseason']  # Target variable

# Define the desired distribution
class_distribution = {
    1: 1,
    2: 1,
    3: 2,
    4: 4,
    5: 8,
    6: 16,
    7: 32
}

# Calculate class weights
total_instances = sum(class_distribution.values())
class_weights = {label: total_instances / (num_instances * 7) for label, num_instances in class_distribution.items()}

# Convert class weights to the required format
class_weights_list = [{label: weight} for label, weight in class_weights.items()]
#class_weights_dict = [{0:1,1:32},{0:1,2:32},{0:1,3:16},{0:1,4:8},{0:1,5:4},{0:1,6:2},{0:1,7:1}]
#print(class_weights_dict, class_weights)
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Random Forest Classifier
rf_classifier = RandomForestClassifier(n_estimators=1000, random_state=42, class_weight=class_weights, min_samples_split=20)

# Train the Random Forest Classifier
rf_classifier.fit(X_train, y_train)

# Make predictions
y_pred = rf_classifier.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

preds_2024 = rf_classifier.predict(analysis_2024_madness_df)
preds_2024_df = pd.DataFrame({'Predictions_2024': preds_2024})

print(preds_2024_df, preds_2024_df.value_counts())





