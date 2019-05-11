#to obtain personality of CS Engineers

from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import tweepy
#import nltk 
#nltk.download('punkt')Punkt Sentence Tokenizer divides a text into a list of sentences 
#nltk.download('stopwords')
import re
import string
from unidecode import unidecode
import csv
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter

ckey='2aVoN4HLmXXyrYP4SY9u3hFdO'
csecret='06esow282t9RAcYNaO0fVFWlL4KcaItZsWuO83oHcUO3mBPJe5'
atoken='1092793465535168512-GDDk4TRBSknA36RbU0bCIU8zX2MSlK'
asecret='pQEUsIKRpoqhRuSCCZUUZhy5zgSnDTpNOgrJeV1gk7Qb1'
auth=tweepy.OAuthHandler(ckey, csecret)
auth.set_access_token(atoken, asecret)
api=tweepy.API(auth)

regex_str = [
    r'<[^>]+>',  # HTML tags
    r'(?:@[\w_]+)',  # @-mentions
    r"(?:\#+[\w_]+[\w\'_\-]*[\w_]+)",  # hash-tags
    r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&amp;+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+',  # URLs

    r'(?:(?:\d+,?)+(?:\.?\d+)?)',  # numbers
    r"(?:[a-z][a-z'\-_]+[a-z])",  # words with - and '
    r'(?:[\w_]+)',  # other words
    r'(?:\S)'  # anything else
]

tokens_re = re.compile(r'(' + '|'.join(regex_str) + ')', re.VERBOSE | re.IGNORECASE)

def tokenize(s):
    return tokens_re.findall(s)

def preprocess(s, lowercase=False):
    tokens = tokenize(s)
    return tokens

def preproc(s):
	s= unidecode(s)
	POSTagger=preprocess(s)
	#print(POSTagger)

	#tweet=' '.join(POSTagger)
	stop_words = set(stopwords.words('english'))
	filtered_sentence = []
	for w in POSTagger:
	    if w not in stop_words:
	        filtered_sentence.append(w)
	#print(filtered_sentence)
	stemmed_sentence=[]
	stemmer2 = SnowballStemmer("english", ignore_stopwords=True)
	for w in filtered_sentence:
		stemmed_sentence.append(stemmer2.stem(w))
	#print(stemmed_sentence)

	temp = ' '.join(c for c in stemmed_sentence if c not in string.punctuation) 
	preProcessed=temp.split(" ")
	final=[]
	for i in preProcessed:
		if i not in final:
			if i.isdigit():
				pass
			else:
				if 'http' not in i:
					final.append(i)
	temp1=' '.join(c for c in final)
	return temp1

def getTweets(user):
	csvFile = open('csengineerstweets.csv', 'a', newline='')
	csvWriter = csv.writer(csvFile)
	try:
		for i in range(0,6):
			tweets=api.user_timeline(screen_name = user, count = 1000, include_rts=True, page=i)
			for status in tweets:
				tw=preproc(status.text)
				if tw.find(" ") == -1:#blank line
					tw="blank"
				csvWriter.writerow([tw])
	except tweepy.TweepError:
		print("Failed to run the command on that user")
	csvFile.close()


print("Myers-Briggs Personality Indicator for CS-ENGINEERS: E-Extraversion/I-Introversion S-Sensing/N-Intuition T-Thinking/F-Feeling J-Judgement/P-Perception")

account=["@NathanBenaich","@j2bryson","@alexjc","@soumithchintala","@adampaulcoates","@cybercodetwins",
         "@jeggers","@etzioni","@MFordFuture","@ageitgey","@johnchavens","@jjkardwell","@davidwkenny",
         "@tessalau","@petitegeek","@LearnOpenCV","@GaryMarcus","@chrismessina","@elonmusk",
         "@tweetsoutloud"]

for j in range(0,20):
    print(j)
    username=account[j]
    getTweets(username)
with open('csengineerstweets.csv','rt') as f:
	csvReader=csv.reader(f)
	tweetList=[rows[0] for rows in csvReader]#list of tweets

with open('newfrequency300.csv','rt') as f:
	csvReader=csv.reader(f)
	mydict={rows[1]: int(rows[0]) for rows in csvReader}#creating dict from csv file
#print(mydict)
vectorizer=TfidfVectorizer(vocabulary=mydict,min_df=1)
x=vectorizer.fit_transform(tweetList).toarray()
df=pd.DataFrame(x)

model_IE = pickle.load(open("IEFinal.sav", 'rb'))
model_SN = pickle.load(open("SNFinal.sav", 'rb'))
model_TF = pickle.load(open('TFFinal.sav', 'rb'))
model_PJ = pickle.load(open('PJFinal.sav', 'rb'))

answer=[]
IE=model_IE.predict(df)
SN=model_SN.predict(df)
TF=model_TF.predict(df)
PJ=model_PJ.predict(df)

b = Counter(IE)
value=b.most_common(1)
print(value)
if value[0][0] == 0.0:
	answer.append("I")
else:
	answer.append("E")

b = Counter(SN)
value=b.most_common(1)
print(value)
if value[0][0] == 1.0:
	answer.append("S")
else:
	answer.append("N")

b = Counter(TF)
value=b.most_common(1)
print(value)
if value[0][0] == 1.0:
	answer.append("T")
else:
	answer.append("F")

b = Counter(PJ)
value=b.most_common(1)
print(value)
if value[0][0] == 0.0:
	answer.append("P")
else:
	answer.append("J")
mbti=" ".join(answer)
print(mbti)
