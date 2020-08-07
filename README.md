# topic_modelling

Topic modelling with LDA.   
LDA assumes that documents consists of a mixture of topics. Those topics then generate words based on their probability distribution. Given a dataset of documents, LDA backtracks and tries to figure out what topics would create those documents in the first place.   
Refer the following for detailed explanation about working of LDA model.   

https://www.analyticsvidhya.com/blog/2016/08/beginners-guide-to-topic-modeling-in-python/?#   
 

## Cleaning and Preprocessing

```ruby
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
## Importing data data from a bunch of documents

```ruby
combined_words = ""
docs = []
for transcript_file_name in glob.iglob('./transcripts/train//*.*', recursive=True):
    #print(os.path.basename(transcript_file_name))
    data = open(transcript_file_name).readlines()
    speaker_data = {line.split(":")[0]:line.split(":")[1] for line in data}
    words_in_file = ""
    speaker_dic ={}
    for name,words in  speaker_data.items():
        words = words.replace("\n","").lower()
        words_in_file = words_in_file + words
        if name.split("_")[0] in speaker_dic:
            speaker_dic[name.split("_")[0]] += words
        else:
            speaker_dic[name.split("_")[0]] = words
    #print("Number of words in the file :",str(len(words_in_file)))
    combined_words += words_in_file
    docs.append([words_in_file])
 ```
    
## Preparing data for LDA model

```ruby
cleaned_docs = []
for doc in docs:
    for word in doc:
        cd = preprocess(word)
        cleaned_docs.append(cd)
```
## Preparing dictionary,document-term-matrix for LDA implementaton and implementing LDA model
```ruby
dictionary = gensim.corpora.Dictionary(cleaned_docs)
dictionary.filter_extremes(no_below=1, no_above=0.5, keep_n=100000) # optional
bow_corpus = [dictionary.doc2bow(doc) for doc in cleaned_docs]
ldamodels = gensim.models.ldamodel.LdaModel(bow_corpus, num_topics = 4, id2word=dictionary, passes=30)
```
## Printing the output
```ruby
for i in ldamodels.print_topics(num_words = 18): 
    for j in i: print (j)
```
## Using pyLDAvis to visualize
```ruby
import pyLDAvis.gensim
pyLDAvis.enable_notebook()
vis = pyLDAvis.gensim.prepare(ldamodels, bow_corpus, dictionary=ldamodels.id2word)
vis
```

## Testing on unseen document
```ruby
unseen_doc_file_path = './transcripts/test/unseen_transcript.txt'
combined_words = ""
docs = []
data = open(unseen_doc_file_path).readlines()
speaker_data = {line.split(":")[0]:line.split(":")[1] for line in data}
words_in_file = ""
speaker_dic ={}
for name,words in  speaker_data.items():
    words = words.replace("\n","").lower()
    words_in_file = words_in_file + words
    if name.split("_")[0] in speaker_dic:
        speaker_dic[name.split("_")[0]] += words
    else:
        speaker_dic[name.split("_")[0]] = words
    #print("Number of words in the file :",str(len(words_in_file)))
combined_words += words_in_file
docs.append([words_in_file])
cleaned_docs = []
for doc in docs:
    for word in doc:
        cd = preprocess(word)
        cleaned_docs.append(cd)
        
bow_vector = dictionary.doc2bow(cleaned_docs[0])

for index, score in sorted(ldamodels[bow_vector], key=lambda tup: -1*tup[1]):
    print("Score: {}\t Topic: {}".format(score, ldamodels.print_topic(index, 7)))   
```

        
