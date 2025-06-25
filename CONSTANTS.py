import os

COIN = 'BTC'
EMPTY_STRING = '-'
CMC_KEY = os.getenv('CMC_KEY', 'Set your CMC API key in the website')
LIMIT = 365
TRAINING_COLUMNS = 'training_columns.txt'
TRAIN_PCT = 0.8
MODEL = "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis"
PATH = f'newspapers/{COIN}_newspapers.csv'
SLEEP = 3
SERPAPI_KEY = os.getenv('SERPAPI_KEY', 'Set your SERPAPI API key in the website')
FILL = -99999999.0

TEST_DAYS = 7