from ast import keyword
from dataclasses import dataclass
import tweepy
import configparser
import pandas as pd 

#read configs
config = configparser.ConfigParser()
config.read('config.ini')

api_key = config['tweeter']['api_key']
api_key_secret = config['tweeter']['api_key_secret']
access_token = config['tweeter']['access_token']
access_token_secret = config['tweeter']['access_token_secret']

# authentication 
auth = tweepy.OAuthHandler(api_key, api_key_secret)
auth.set_access_token(access_token, access_token_secret)
api =tweepy.API(auth)
public_tweets = api.home_timeline()

###Writes last 20 twits in file 
for tweet in public_tweets:
  print(tweet.text)
print(public_tweets[0].user.screen_name)
columns = ['Time', 'User', 'Tweet']
data = []
for tweet in public_tweets:
    data.append([tweet.created_at, tweet.user.screen_name, tweet.text])
df = pd.DataFrame(data, columns=columns)
df.to_csv('tweets.csv')


###Getting some tweets of a user
user = 'SpaceX'
limit = 30

tweets = api.user_timeline(screen_name = user, count = limit, tweet_mode = 'extended')
for tweet in tweets:
    print(tweet.full_text)
    



