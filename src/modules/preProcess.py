import pandas as pd
import numpy as np
import io
import re
import string
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
import nltk
import spacy
from spacy.lang.bg.stop_words import STOP_WORDS as BG_STOPWORDS

DELIMITER = '//'

# for display
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


# Load the data
with io.open('/Users/kyosuke.morita/project_v2/Lyrics_generator/resources/data/krisko_lyrics.txt', encoding='utf-8') as f:
    krisko_text = f.read().lower().replace('\n', ' \n ')
# songs = text.split(DELIMITER) -- I can do sentiment analysis on each songs
print('Corpus length in characters:', len(krisko_text))

words = [w for w in krisko_text.split(' ') if w.strip() != '' or w == '\n']
print('Corpus length in words:', len(words))

word_df = pd.DataFrame(words,columns=['word'])
word_df.word.nunique()
top = word_df.word.value_counts().head(100)

# Preprocessing
krisko_cleaned = re.sub('\[,.*?“”…\]', '', krisko_text)

# tokenise
TOKENS = word_tokenize(krisko_cleaned) 

# add some new stop words
EXTRA_STOPWORDS = {'теб','дон','кво','к\'во','бях','мене','нашият','ма','ше','yeah',
                    'недей','ей','ко','bang','ам','тебе','you','тука','мойта','тва',
                    'але-але-алелуя','алеалеалелуя','кат','tak','“','моа','оп','о',
                    '’','ся','та','тез','дето','ја','aз','tik','i','ѝ','ток','твоя',
                    'a','some','ideal','petroff','–','так','кво','дай','тия','ee','к'}
BG_STOPWORDS.update(EXTRA_STOPWORDS)

filtered_sentence = []
  
for w in TOKENS: 
    if w not in BG_STOPWORDS: 
        filtered_sentence.append(w)

with open ('/Users/kyosuke.morita/project_v2/Lyrics_generator/resources/data/krisko_cleaned.txt','w') as output:
    output.write(str(filtered_sentence))

with io.open('/Users/kyosuke.morita/project_v2/Lyrics_generator/resources/data/krisko_cleaned.txt', encoding='utf-8') as f:
    krisko_cleaned = f.read()

krisko_cleaned = re.sub('[%s]' % re.escape(string.punctuation), '', krisko_cleaned)

words = [w for w in krisko_cleaned.split(' ') if w.strip() != '' or w == '\n']
print('Corpus length in words:', len(words))

word_df = pd.DataFrame(words,columns=['word'])
word_df.word.nunique()
top = word_df.word.value_counts().head(50)