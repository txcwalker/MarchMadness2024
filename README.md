# MarchMadness2024
A couple of different models to try and predict the results of the March Madness College basketball tournament for 2024

There are two different models version of the model included, both are based off of the same data (mostly from Kenpom.com, but also some of the data was provided by Kaggle).
The first model is blind. It does not know which teams are playing each other and just predicts a raw percentage for them to advance to a specified round. Take this year for example. The model rated Uconn, Auburn and and Iowa St. as three of the five most likely teams to win the tournament. However all of these teams were in the same region so only one of them could evenmake the final four let alone win.

The second model is a simulation of the tournament X times. A table is created with every possibe matchup in the tournament. Each team is given a percentage chance to win each matchup. Then each game is simulated one at a time based on the table probabilities. A random number is selected between 0 and 1. Say Team A has a 65% chance to beat Team B based on the table. If the randomly generated number <.65 Team A advances, otherwise Team B does. There are 63 games per tournament and the tournmaent can be simulated X amount of times. After the simulations are complete a table is produced showing what percentage of the time each team made it every round of the tournament.

# Models
# Model 1 
The first model was surprisingly great. While it missed on teams Samford and New Mexico. It was all aboard the NC state train and completely off of UNC. 
The Following Files are for the first model and the rest of the files are for the second: cbb.csv, cbb13.csv, cbb24.csv, MTeams.csv, probs_merged_2024.csv, tournament_teams_2013.csv and model.py
The results file is probs_merged_2024.csv

# Model 2 
The second model... needs improvement. The results of the sim table are included in this repo and if you take a look you can see that the seed matchups with huge disparities, the model is not very aggressive. We would expect Uconn (1) to beat Stetson (16) Nearly 100% of the time but the model is no where near this number. 
For this model the results file is _round_counts_2024.csv. This is not what would have been submitted to Kaggle just a easy to read summary of what 10000 simulations produced. 
Additional for this model not all of the data files were included since a few of them were massive. However they can all be downloaded and referenced here: https://www.kaggle.com/competitions/march-machine-learning-mania-2024/data

# Kaggle 
This was done for the competition hosted on Kaggle every year. I did not submit my code because I was unable to find the same stats I used for the Mens as the Womens (Predictions for both tournaments were needed in order for a submission to be considered valid). In fact I found it difficult to find any statistics that were easily downloadable for the Womens game

#Borrowed Code
One note is a portion of the simulation code was taken (with permission) from a Kaggle user. I modified it a bit to work with data. 
