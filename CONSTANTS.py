COIN = 'BTC'

EMPTY_STRING = '-'
CMC_KEY = "98464488-2db9-4ddc-9b98-6b48f8b623dc"
LIMIT = 365
TRAINING_COLUMNS = 'training_columns.txt'
TRAIN_PCT = 0.8
MODEL = "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis"
PATH = f'newspapers/{COIN}_newspapers.csv'
SLEEP = 3
SERPAPI_KEY = "d8b42ca90077ec59802d53ad3d55e7e05d38ebac54426fbfa8913b8f40f68e24"
FILL = -99999999.0

# Early Stopping Constants
LOSS_MULTIPLIER = 1
LOSS_10E_THING = -8
LOSS_THRESHOLD = LOSS_MULTIPLIER * 10^LOSS_10E_THING

START = '2009-01-01'
END = '2023-10-01'