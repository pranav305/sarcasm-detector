import pandas as pd
import pickle

filename = "train-balanced-sarcasm.csv"

df = pd.read_csv(filename)


all_data = df.drop_duplicates(keep='first', inplace=False)

cleaned_data = all_data.dropna()

sentences = cleaned_data['comment']

y = cleaned_data['label']

from sklearn.model_selection import train_test_split # imports module from package

x_train, x_test, y_train, y_test = train_test_split(sentences, y, test_size=0.25, random_state=1000)

from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()

cv = vectorizer.fit(x_train)
pickle.dump(cv, open("cv.pickle","wb"))

X_train = vectorizer.transform(x_train)

X_test  = vectorizer.transform(x_test)

from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression()

lr = classifier.fit(X_train, y_train)

final_model_filename = 'lr.pickle'
pickle.dump(lr, open(final_model_filename,'wb'))

score = classifier.score(X_test, y_test)

print("\n Accuracy:", score)   #model accuracy

while True:
    userInput = input("Enter a comment, to check for sarcasm \n >>  ")
    
    word = [userInput]
    test_data = vectorizer.transform(word)
    
    
    result = classifier.predict(test_data)
    prob = classifier.predict_proba(test_data)
    print(result)
    print(prob)
    
    continue