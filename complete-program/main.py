from sklearn.linear_model import LogisticRegression
import pickle
import warnings

warnings.simplefilter('ignore')

vectorizer = pickle.load(open('cv.pickle','rb'))

classifier = LogisticRegression()

final_model_filename = 'lr.pickle'
loaded_model = pickle.load(open(final_model_filename,'rb'))

while True:
    userInput = input("\nEnter a comment, to check for sarcasm \n >>  ")
    
    word = [userInput]
    test_data = vectorizer.transform(word)
    
    
    result = loaded_model.predict(test_data)
    prob = loaded_model.predict_proba(test_data)

  
    if result == [1]:
      ans = f'Sarcastic \n{(prob[0][1] * 100):.1f}% confidence'
    elif result == [0]:
      ans = f'Not Sarcastic \n{(prob[0][0] * 100):.1f}% confidence'

    print(ans)

        
    continue