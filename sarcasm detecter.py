import pickle
from sklearn.linear_model import LogisticRegression

vectorizer = pickle.load(open('cv.pickle','rb'))

classifier = LogisticRegression()

final_model_filename = 'lr.pickle'
loaded_model = pickle.load(open(final_model_filename,'rb'))

while True:
    userInput = input("Enter a comment, to check for sarcasm \n >>  ")
    
    word = [userInput]
    test_data = vectorizer.transform(word)
    
    
    result = loaded_model.predict(test_data)
    prob = loaded_model.predict_proba(test_data)
    print(result)
    print(prob)
    
    continue