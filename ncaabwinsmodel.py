import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import random
import seaborn as sns
from sklearn.model_selection import cross_val_score

#setting np random seed
np.random.seed(42)
#https://www.kaggle.com/competitions/march-machine-learning-mania-2024/data
#imporing data
#Teams
mteams = pd.read_csv("data/MTeams.csv")
wteams = pd.read_csv('data/WTeams.csv')

#Renaming Team Column for both
mteams = mteams.rename(columns = {'TeamName':'Team'})
wteams = wteams.rename(columns = {'TeamName':'Team'})

#kenpom
kenpom_raw_df = pd.read_csv('data/kenpom_raw.csv')

#Spliting Team to get tournament seed and dropping ranks for all other stats
kenpom_df = kenpom_raw_df.drop(['Rk', 'Unnamed: 6','Unnamed: 8','Unnamed: 10','Unnamed: 12','Unnamed: 14','Unnamed: 16'
                                ,'Unnamed: 18','Unnamed: 20'], axis = 1)

kenpom_df[['Team', 'Seed']] = kenpom_df['Team'].str.extract(r'^(.*?)(\d*)$')
kenpom_df['Team'] = kenpom_df['Team'].str.rstrip()

#Re import kenpom after fixing names
kenpom_fin_df = pd.read_csv('data/kenpom_data.csv')


#Cut 2020 as it was not a tournament year Thx COVID
#kenpom_fin_df = kenpom_fin_df[kenpom_fin_df['Year'] != 2020]


"""#Debugging
print(kenpom_fin_df['Team'].unique())
# Convert the lists to sets
excel_schools_set = set(kenpom_fin_df['Team'].unique())
teamname_schools_set = set(mteams['Team'].unique())

# Find the schools present in the Excel list but not in the TeamName list
schools_only_in_excel = excel_schools_set - teamname_schools_set

# Find the schools present in the TeamName list but not in the Excel list
schools_only_in_teamname = teamname_schools_set - excel_schools_set

# Print the results
print("Schools present in the Excel list but not in the TeamName list:")
for school in schools_only_in_excel:
    print(school)

print("\nSchools present in the TeamName list but not in the Excel list:")
for school in schools_only_in_teamname:
    print(school)"""

#merge with MTeams to add Team ID to kenpom
kenpom_fin_df = kenpom_fin_df.merge(mteams, how = 'inner', on = 'Team')
kenpom_fin_df = kenpom_fin_df.rename(columns = {'AdjEM.1':'AdjEM_SOS','AdjEM.2':'AdjEM_NCSOS', 'Year':'Season'})
kenpom_fin_df = kenpom_fin_df.drop(columns = 'Seed')

#Importing Seasons, Tournament seeds and results
#Seasons
mseasons_df = pd.read_csv('data/MSeasons.csv')
wseasons_df = pd.read_csv('data/WSeasons.csv')

#Seeds
mseeds_df = pd.read_csv('data/MNCAATourneySeeds.csv')
wseeds_df = pd.read_csv('data/WNCAATourneySeeds.csv')

#Regular season results - compact
mregresultscomp_df = pd.read_csv('data/MRegularSeasonCompactResults.csv')
wregresultscomp_df = pd.read_csv('data/WRegularSeasonCompactResults.csv')

#Tournament Results - compact
mmadresultscomp_df = pd.read_csv('data/MNCAATourneyCompactResults.csv')
wmadresultscomp_df = pd.read_csv('data/WNCAATourneyCompactResults.csv')

#Regular season results - detailed
mregresultsdet_df = pd.read_csv('data/MRegularSeasonDetailedResults.csv')
wregresultsdet_df = pd.read_csv('data/WRegularSeasonDetailedResults.csv')

#Tournament Results - detailed
mmadresultsdet_df = pd.read_csv('data/MNCAATourneyDetailedResults.csv')
wmadresultsdet_df = pd.read_csv('data/WNCAATourneyDetailedResults.csv')

#Coaches
mcoaches_df = pd.read_csv('data/MTeamCoaches.csv')

#Conferences
conf_df = pd.read_csv('data/Conferences.csv')
mconf_df = pd.read_csv('data/MTeamConferences.csv')
wconf_df = pd.read_csv('data/WTeamConferences.csv')

#Conference Tournament Results and Non March Madness tournament teams + results
mconftourn_df = pd.read_csv('data/MConferenceTourneyGames.csv')
msectournteams_df = pd.read_csv('data/MSecondaryTourneyTeams.csv')
msectournres_df = pd.read_csv('data/MSecondaryTourneyCompactResults.csv')

#Tournament Slots
mtournslot_df = pd.read_csv('data/MNCAATourneySlots.csv')
wtournslot_df = pd.read_csv('data/WNCAATourneySlots.csv')
mseedslot_df = pd.read_csv('data/MNCAATourneySeedRoundSlots.csv')

#Kenpom data only goes to 2003 so we are droping anything before
# List of all DataFrames
dataframes = [mseasons_df, wseasons_df, mseeds_df, wseeds_df,
              mregresultscomp_df, wregresultscomp_df, mmadresultscomp_df,
              wmadresultscomp_df, mregresultsdet_df, wregresultsdet_df,
              mmadresultsdet_df, wmadresultsdet_df, mcoaches_df, conf_df,
              mconf_df, wconf_df, mconftourn_df, msectournteams_df,
              msectournres_df, mtournslot_df, mseedslot_df]

#Removing years not in kenpom data
for df in dataframes:
    # Check if 'Year' or 'Season' column exists
    if 'Year' in df.columns:
        # Drop rows where 'Year' is before 2003
        df.drop(df[(df['Year'] < 2003)].index, inplace=True)
    elif 'Season' in df.columns:
        # Drop rows where 'Season' is before 2003
        df.drop(df[(df['Season'] < 2003)].index, inplace=True)



#Copying winning Team to new column and switching everything to
mregresultscomp_df.reset_index(inplace=True)
mregresultscomp_df['Winner'] = mregresultscomp_df['WTeamID']

#Seperating the TeamIDs for Merge
#Need to merge kenpom data for each team. IN order to accomplish the oringal df is being split on TeamID, merged by team
#with kenpom so that each on row will be a game with each team and their respective kenpom stats
wgames_df = mregresultscomp_df[['index','Season','DayNum', 'WTeamID','WScore','WLoc', 'NumOT','Winner']]
lgames_df = mregresultscomp_df[['index','Season','DayNum', 'LTeamID','LScore','WLoc', 'NumOT']]

#Renaming TeamID Columns to Team ID, will be renamed to TeamID_A and TeamID_B on final merge
wgames_df = wgames_df.rename(columns = {'WTeamID':'TeamID'})
lgames_df = lgames_df.rename(columns = {'LTeamID':'TeamID'})

#Merging with kenpom
wgames_df = wgames_df.merge(kenpom_fin_df, on =['TeamID','Season'])
lgames_df = lgames_df.merge(kenpom_fin_df, on =['TeamID','Season'])

#Recreating for Tournament games
#Copying winning Team to new column and switching everything to
mmadresultsdet_df['Winner'] = mmadresultsdet_df['WTeamID']

#Seperating the TeamIDs for Merge
wgamestourn_df = mmadresultsdet_df[['Season','DayNum', 'WTeamID','WScore','WLoc', 'NumOT','Winner']]
lgamestourn_df = mmadresultsdet_df[['Season','DayNum', 'LTeamID','LScore','WLoc', 'NumOT']]

#Renaming TeamID Columns to Team ID, will be renamed to TeamID_A and TeamID_B on final merge
wgamestourn_df = wgamestourn_df.rename(columns = {'WTeamID':'TeamID'})
lgamestourn_df = lgamestourn_df.rename(columns = {'LTeamID':'TeamID'})

#Merging with kenpom
wgamestourn_df = wgamestourn_df.merge(kenpom_fin_df, on =['TeamID','Season'])
lgamestourn_df = lgamestourn_df.merge(kenpom_fin_df, on =['TeamID','Season'])


#Merging everthing together
games_df = wgames_df.merge(lgames_df, on = 'index' ,suffixes = ('_A', '_B'))
gamestourn_df = wgamestourn_df.merge(lgamestourn_df, left_index = True, right_index = True ,suffixes = ('_A', '_B'))

#Final df form for analysis
allgames_df = pd.concat([games_df, gamestourn_df], axis = 0)

#Prepping Everything for analysis
allgames_df = allgames_df.drop(columns = ['W-L_A','W-L_B','index','Season_B', 'DayNum_B','WLoc_B','NumOT_B'])
allgames_df = allgames_df.rename(columns = {'Season_A':'Season', 'DayNum_A':'DayNum','WLoc_A':'WLoc','NumOT_A':'NumOT'})

#The below idea didnt really work. The Teams were assigned home and away based on the WLoc variable
"""# Function to randomly assign home and away teams
def assign_home_away(teams):
    return random.sample(teams, k=2)

# Function to create new columns based on WLoc
def create_home_away_columns(row):
    if row['WLoc'] == 'H':
        row['HomeID'] = row['TeamID_A']
        row['AwayID'] = row['TeamID_B']
        row['HomeScore'] = row['WScore']
        row['AwayScore'] = row['LScore']
    elif row['WLoc'] == 'A':
        row['HomeID'] = row['TeamID_B']
        row['AwayID'] = row['TeamID_A']
        row['HomeScore'] = row['LScore']
        row['AwayScore'] = row['WScore']
    else:  # WLoc == 'N'
        home_team, away_team = assign_home_away([row['TeamID_A'], row['TeamID_B']])
        row['HomeID'] = home_team
        row['AwayID'] = away_team
        if home_team == row['TeamID_A']:
            row['HomeScore'] = row['WScore']
            row['AwayScore'] = row['LScore']
        else:
            row['HomeScore'] = row['LScore']
            row['AwayScore'] = row['WScore']
    return row

# Apply function to each row
allgames_df[['HomeID', 'AwayID', 'HomeScore', 'AwayScore']] = allgames_df.apply(create_home_away_columns, axis=1)[['HomeID', 'AwayID', 'HomeScore', 'AwayScore']]
allgames_df['Winners'] = allgames_df.apply(lambda row: 1 if row['HomeScore'] > row['AwayScore'] else 0, axis=1)
allgames_df.to_csv('data/allgames_df.csv', index=False)

allgames_df = pd.read_csv('data/allgames_df.csv')"""

#Dropping more columns
allgames_df = allgames_df.drop(columns = ['Team_A', 'Team_B','WLoc','WScore','LScore','NumOT','DayNum','Winner']) #'HomeID','AwayID', 'HomeScore', 'AwayScore',

#Conference dictonary
conf_mapping = {
    'SEC': 'sec',
    'CUSA': 'cusa',
    'MAC': 'mac',
    'B12': 'big_twelve',
    'B10': 'big_ten',
    'MWC': 'mwc',
    'BSky': 'big_sky',
    'ASun': 'a_sun',
    'MVC': 'mvc',
    'BE': 'big_east',
    'Horz': 'horizon',
    'OVC': 'ovc',
    'ACC': 'acc',
    'P10': 'pac_ten',
    'Slnd': 'southland',
    'A10': 'aac',
    'SB': 'sun_belt',
    'Ivy': 'ivy',
    'WCC': 'wcc',
    'WAC': 'wac',
    'CAA': 'caa',
    'Pat': 'patriot',
    'MAAC': 'maac',
    'NEC': 'nec',
    'AE': 'aec',
    'SWAC': 'swac',
    'MEAC': 'meac',
    'Sum': 'summit',
    'P12': 'pac_twelve',
    'Amer': 'americ_east'
}
# Apply conference mapping to Conf_A and Conf_B columns
allgames_df['Conf_A'] = allgames_df['Conf_A'].map(conf_mapping)
allgames_df['Conf_B'] = allgames_df['Conf_B'].map(conf_mapping)

# Create LabelEncoder object (for conferences)
label_encoder = LabelEncoder()

# Fit and transform Conf_A and Conf_B columns to integers
allgames_df['Conf_A'] = label_encoder.fit_transform(allgames_df['Conf_A'])
allgames_df['Conf_B'] = label_encoder.fit_transform(allgames_df['Conf_B'])

#Function randomly chosen between provided columns for value in same row
def random_id(row):
    return np.random.choice([row['TeamID_A'], row['TeamID_B']])

#Create the new column 'RandID'
allgames_df['RandID'] = allgames_df.apply(random_id, axis=1)

#Makes sure that the first set of stats is always associated with the RandID team and the second set is their opponent
def stat_swap(df):
    #Empty list to put results in
    selected_rows = []

    #Columns no associated with kenpom and always neeeded
    static_columns = ['TeamID_A', 'TeamID_B', 'Season', 'RandID']
    for index, row in df.iterrows():

        #Get column suffixes
        columns_A = [col for col in df.columns if col.endswith('_A')]
        columns_B = [col for col in df.columns if col.endswith('_B')]

        #Setting up column order based on which Team was chosen for RandID
        if row['RandID'] == row['TeamID_A']:
            selected_columns = static_columns + columns_A + columns_B
        else:
            selected_columns = static_columns + columns_B + columns_A

        #Getting data in form for eventual move to df
        selected_data = row[selected_columns].to_dict()
        selected_rows.append(selected_data)

    selected_df = pd.DataFrame(selected_rows)
    return selected_df

#Applying stat swap
allgames_df2 = stat_swap(allgames_df)

#Creating variable to be dependent variable in model
allgames_df2['Winner'] = (allgames_df2['TeamID_A'] == allgames_df2['RandID']).astype(int)

#Dropping both TeamIDs so that model does not see that TeamID_A is always the winner
allgames_df2 =allgames_df2.drop(columns = ['TeamID_A', 'TeamID_B'])

#Saving final df so it can be visualized easier in case of debugging
allgames_df2.to_csv('data/_allgames2.csv', index = False)

#Creating the Random Forest Model to be used in the Sims
#Defining features and target
X = allgames_df2.drop(['Winner'], axis=1)
y = allgames_df2['Winner']

#Test train splot
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Initialize RandomForestClassifier
nccabwins = RandomForestClassifier(n_estimators=100, random_state=42)

#Train the model
nccabwins.fit(X_train, y_train)

#Predict on the test set
y_pred = nccabwins.predict(X_test)

#Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

#Feature Importance
#Get feature importances
importance = nccabwins.feature_importances_

# Sort feature importance in descending order
indices = np.argsort(importance)[::-1]

# Rearrange feature names so they match the sorted feature importance
sorted_feature_names = [X.columns[i] for i in indices]

# Plot
plt.figure(figsize=(10, 6))
plt.title("Feature Importances")
plt.bar(range(X.shape[1]), importance[indices], align="center")
plt.xticks(range(X.shape[1]), sorted_feature_names, rotation=90)
plt.xlabel("Feature")
plt.ylabel("Importance")
plt.tight_layout()
#plt.show()

#Confusion Matrix
# Making predictions on the test data
predictions = nccabwins.predict(X_test)

#Calculating the confusion matrix
conf_matrix = confusion_matrix(y_test, predictions)

#Plotting the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='d')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
#plt.show()









