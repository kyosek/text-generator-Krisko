from nltk.tokenize import sent_tokenize, word_tokenize
from gensim.models import Word2Vec 
from gensim.models import Phrases


# preparation
input_words = []

for i in sent_tokenize(krisko_cleaned): 
    temp = [] 
      
    # tokenize the sentence into words 
    for j in word_tokenize(i): 
        temp.append(j.lower()) 
  
    input_words.append(temp)

# Create a word2vec model
w2vModel = Word2Vec(input_words, min_count=1, size=50, window=1)

w2vModel.similarity('мъж','жена')
pd.DataFrame(w2vModel.most_similar('жена'),columns=['similar_word','similarity'])
pd.DataFrame(w2vModel.most_similar('мъж'),columns=['similar_word','similarity'])

w2vModel.similarity('искам','искаш')
pd.DataFrame(w2vModel.predict_output_word('искам',30),columns=['next_word','probability'])
pd.DataFrame(w2vModel.predict_output_word('имаш',30),columns=['next_word','probability'])

w2vModel.predict_output_word('аз сум')

# Create a skip gram model
sgModel = Word2Vec(input_words, min_count=1, size=50, window=3, sg=1)

sgModel.similarity('искам','искаш')

pd.DataFrame(sgModel.most_similar('искам'),columns=['similar_word','similarity'])
pd.DataFrame(sgModel.most_similar('искаш'),columns=['similar_word','similarity'])
pd.DataFrame(sgModel.predict_output_word('искам'),columns=['similar_word','similarity'])
pd.DataFrame(sgModel.predict_output_word('искаш'),columns=['similar_word','similarity'])