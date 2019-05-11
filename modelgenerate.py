import csv
import pandas
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import GaussianNB

csvFile=open('newfrequency300.csv', 'rt')
csvReader=csv.reader(csvFile)
mydict={row[1]: int(row[0]) for row in csvReader} #creating dict from csv file

y=[]
with open ('PJFinaltest.csv', 'rt') as f:
	reader=csv.reader(f)
	corpus=[rows[0] for rows in reader]#getting corpus data in a list

with open ('PJFinaltest.csv', 'rt') as f:
	csvReader1=csv.reader(f)
	for rows in csvReader1:
		y.append([int(rows[1])]) #getting corpus values in a list

vectorizer=TfidfVectorizer(vocabulary=mydict,min_df=1)
x=vectorizer.fit_transform(corpus).toarray()
result=np.append(x,y,axis=1)
X=pandas.DataFrame(result)
#print(X)
model=GaussianNB()
train = X.sample(frac=0.8, random_state=1)
test=X.drop(train.index)
y_train=train[301]#stores the last column 
y_test=test[301]
print(train.shape)#Return a tuple of the shape of the underlying data(rows X columns)
print(test.shape)
xtrain=train.drop(301,axis=1)#drops last column
xtest=test.drop(301,axis=1)
model.fit(xtrain,y_train)
pickle.dump(model, open('PJFinal.sav', 'wb'))#saves the model in BNPFFinal.sav to be loaded later
del result

y=[]
with open ('IEFinaltest.csv', 'rt') as f:
	reader=csv.reader(f)
	corpus=[rows[0] for rows in reader]

with open ('IEFinaltest.csv', 'rt') as f:
	csvReader1=csv.reader(f)
	for rows in csvReader1:
		y.append([int(rows[1])])
        
vectorizer=TfidfVectorizer(vocabulary=mydict,min_df=1)
x=vectorizer.fit_transform(corpus).toarray()
result=np.append(x,y,axis=1)
X=pandas.DataFrame(result)
model=GaussianNB()
train = X.sample(frac=0.8, random_state=1)
test=X.drop(train.index)
y_train=train[301]#stores the last column 
y_test=test[301]
print(train.shape)#Return a tuple of the shape of the underlying data(rows X columns)
print(test.shape)
xtrain=train.drop(301,axis=1)#drops last column
xtest=test.drop(301,axis=1)
model.fit(xtrain,y_train)
pickle.dump(model, open('IEFinal.sav', 'wb'))
del result

y=[]
with open ('TFFinaltest.csv', 'rt') as f:
	reader=csv.reader(f)
	corpus=[rows[0] for rows in reader]

with open ('TFFinaltest.csv', 'rt') as f:
	csvReader1=csv.reader(f)
	for rows in csvReader1:
		y.append([int(rows[1])])
        
vectorizer=TfidfVectorizer(vocabulary=mydict,min_df=1)
x=vectorizer.fit_transform(corpus).toarray()
result=np.append(x,y,axis=1)
X=pandas.DataFrame(result)
model=GaussianNB()
train = X.sample(frac=0.8, random_state=1)
test=X.drop(train.index)
y_train=train[301]#stores the last column 
y_test=test[301]
print(train.shape)#Return a tuple of the shape of the underlying data(rows X columns)
print(test.shape)
xtrain=train.drop(301,axis=1)#drops last column
xtest=test.drop(301,axis=1)
model.fit(xtrain,y_train)
pickle.dump(model, open('TFFinal.sav', 'wb'))
del result

y=[]
with open ('SNFinaltest.csv', 'rt') as f:
	reader=csv.reader(f)
	corpus=[rows[0] for rows in reader]

with open ('SNFinaltest.csv', 'rt') as f:
	csvReader1=csv.reader(f)
	for rows in csvReader1:
		y.append([int(rows[1])])
        
vectorizer=TfidfVectorizer(vocabulary=mydict,min_df=1)
x=vectorizer.fit_transform(corpus).toarray()
result=np.append(x,y,axis=1)
X=pandas.DataFrame(result)
model=GaussianNB()
train = X.sample(frac=0.8, random_state=1)
test=X.drop(train.index)
y_train=train[301]#stores the last column 
y_test=test[301]
print(train.shape)#Return a tuple of the shape of the underlying data(rows X columns)
print(test.shape)
xtrain=train.drop(301,axis=1)#drops last column
xtest=test.drop(301,axis=1)
model.fit(xtrain,y_train)
pickle.dump(model, open('SNFinal.sav', 'wb'))
del result