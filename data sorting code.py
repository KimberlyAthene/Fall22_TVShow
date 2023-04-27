# ISYE 6740 Project
# This is to consolidate the IMDb data pulled 11/3/22
# For the Project: Predicting Longevity of TV Shows

import numpy as np
import pandas as pd
import os


#All datasets were downloaded from datasets.imdbws.com

# READ IN DATASETS

# base df read from: title.basics.tsv.gz
# this dataset was filtered by titleType = 'tvMiniSeries' and 'tvSeries'
#  and 'genre' was separated into three columns, with the third being deleted (lack of info), and \N
#    was replaced with 'None'
base_df = pd.read_csv('C:\\Users\\Kim\\Documents\\GT Courses\\ISYE 6740 Fall 2022\\Project\\Data\\titlebasics.csv',
                     encoding='latin-1')


# ratings df read from: title.ratings.tsv.gz
ratings_df = pd.read_csv('C:\\Users\\Kim\\Documents\\GT Courses\\ISYE 6740 Fall 2022\\Project\\Data\\ratings.csv',
                          encoding='latin-1')


# episode df read from: title.episode.tsv.gz
# This was sorted by 'parentTconst' and all \N (NA) were filled with 0
episode_df = pd.read_csv('C:\\Users\\Kim\\Documents\\GT Courses\\ISYE 6740 Fall 2022\\Project\\Data\\episode.csv',
                          encoding='latin-1')


# principals df read from: title.principals.tsv.gz
principals_df = pd.read_csv('C:\\Users\\Kim\\Documents\\GT Courses\\ISYE 6740 Fall 2022\\Project\\Data\\principals.csv',
                          encoding='latin-1')

################################################################################################


# EDIT DATAFRAMES TO GET COLUMNS NEEDED FOR ANALYSIS

# Starting with principle_df:
# I think I should take the principals_df and count the types of category per tconst
# and then use unstack to get individual columns by category of job
pcntdf = principals_df.groupby('tconst')['category'].value_counts().unstack().fillna(0)
#print(pcntdf.head)


# Editing episode_df:
# Rename some columns of episode in order to better merge it with the others
# the tconst column of episode_df refers to the episode title, not the show title
eps_df = episode_df.rename(columns={'tconst':'epconst', 'parentTconst':'tconst'})

# Now I need to use the edited episode_df (eps_df) to obtain the maximum season number per tconst
#  and find the average maximum number of episodes per tv show sseasons
eps_df['seasonNumber'] = pd.to_numeric(eps_df.seasonNumber) # change to int for max/mean operations
eps_df['episodeNumber'] = pd.to_numeric(eps_df.episodeNumber)
# find max season per show
eps_df['maxSeason'] = eps_df.groupby('tconst').seasonNumber.max()[eps_df.tconst].reset_index().seasonNumber
# find max episode per season per show
mxeps_df = eps_df.groupby(['tconst','seasonNumber'])['episodeNumber'].max().reset_index().fillna(0) 
# create two series of just the columns that are important
mxseas = eps_df.groupby('tconst')['seasonNumber'].max()
avgmxeps = mxeps_df.groupby('tconst')['episodeNumber'].mean()
# create one dataframe for this and change the column names
edeps_df = pd.concat([mxseas, avgmxeps], axis=1).reset_index()
epis_df = edeps_df.rename({'seasonNumber':'maxSeason','episodeNumber':'avgMaxEps'}, axis=1)
#print(epis_df.head)


# edit ratings_df (just drop the numVotes column because it's too biased to use)
ratings_df.drop('numVotes', axis=1, inplace=True)
#print(ratings_df.head)


# Finally edit base_df (ratings df doesn't need to be edited, we're only keeping avgRatings anyway)
# First drop some unnecessary columns for the analysis
base_df.drop(['primaryTitle', 'originalTitle'], axis=1, inplace=True)
#print(base_df.head)





################################################################################################


# Now I want to merge these four datasets into one so that I only keep the data useful for my problem
# Principals, episode, and ratings are all about the same shape of 1048575 rows (episode has one less row)
# Because base was filtered to only include tv shows, it has 51581 rows, which is still a significant amount 
#   of data points

prin_eps_df = pd.merge(epis_df, pcntdf, on='tconst')
#print(prin_eps_df.head)

base_eps_df = pd.merge(base_df, epis_df, on='tconst')
#print(base_eps_df)

bep_df = pd.merge(base_eps_df, pcntdf, on='tconst')
#print(bep_df)

bepr_df = pd.merge(bep_df, ratings_df, on='tconst')
#print(bepr_df.head)

os.makedirs('C:\\Users\\Kim\\Documents\\GT Courses\\ISYE 6740 Fall 2022\\Project', exist_ok=True)
bepr_df.to_csv('C:\\Users\\Kim\\Documents\\GT Courses\\ISYE 6740 Fall 2022\\Project\\data.csv')