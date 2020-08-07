# topic_modelling

Topic modelling with LDA.   
Also used pyLDAvis to visualise the results of topic modelling.  

# Cleaning and Preprocessing

# importing customized stopwords from customized_stopwords.txt
```
with open ('customized_stopwords', 'rb') as fp:
    customized_stopwords = pickle.load(fp)
more_stop_words = ['finish','start','tomorrow','work','agree','think','middle','dicide','write','haven','understand','print','call','return','talk','happen']   
customized_stopwords=customized_stopwords + more_stop_words
#stemmer = SnowballStemmer("english")
def lemmatize(word):                                    # input is a word that is to be converted to root word for verb
    return WordNetLemmatizer().lemmatize(word, pos = 'v')

def preprocess(text):
    result=[]
    for token in gensim.utils.simple_preprocess(text) :
        if (token not in gensim.parsing.preprocessing.STOPWORDS) and (len(token) > 4) and (token not in customized_stopwords):
            if token not in customized_stopwords:
                
                result.append(lemmatize(token))
                if token == 'happen':
                    print("yy")
            
    return result
```
