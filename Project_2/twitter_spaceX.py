from ast import keyword
from dataclasses import dataclass
import tweepy
import configparser
import pandas as pd 

#read configs
#config = configparser.ConfigParser()
#config.read('config.ini')

#api_key = config['tweeter']['api_key']
#api_key_secret = config['tweeter']['api_key_secret']
#access_token = config['tweeter']['access_token']
#access_token_secret = config['tweeter']['access_token_secret']

api_key = 'BrtedvLydbXZWQxaLjNK3hxEe'
api_key_secret = 'cDI1APaFrIAsJ4jFDD4YQUA6VCojzkaKZDLLwtrnhNcQztRPPm'
bearer_token = 'AAAAAAAAAAAAAAAAAAAAAI6aiQEAAAAALj2dFRoHhFumZjbxTYjJdWzr34Y%3DnBkF3S2rg644mjlNYMOAgSxHlxrjd4jfr5mszMXTGSeLYt4ez2'
access_token = '1247052156248756224-a35VQOmCztKTkStuap7tAVeNliz0Aq'
access_token_secret = 'U9tDATQr2wZ95WLtZRSq6D0PdzKN1z91PyCHPqMe5j1a5'
# authentication 
api = tweepy.Client(bearer_token=bearer_token, consumer_key= api_key,consumer_secret= api_key_secret,access_token= access_token,access_token_secret= access_token_secret)

user = 'Falcon 9\'s launch from:SpaceX'
max_limit = 10
tweets = api.search_recent_tweets( query=user ,tweet_fields=['context_annotations', 'created_at'],max_results = max_limit,user_auth=False)

for tweet in tweets.data:
    print(tweet.text)
    print('\n')
    #print(tweet.created_at)

