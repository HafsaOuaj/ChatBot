# required modules
import random
import json
import pickle
import numpy as np
import nltk
from keras.models import load_model
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

# loading the files we made previously
intents = json.loads(open("intense.json").read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('hafsabotmodel.h5')


# Format the sentence we gave as input 
def clean_up_sentences(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) 
                      for word in sentence_words]
    return sentence_words

# bagw(sentence): This function will append 1 to a list variable 
# ‘bag’ if the word is contained inside our input and
# is also present in the list of words created earlier.


def bagw(sentence):
	
	# separate out words from the input sentence
	sentence_words = clean_up_sentences(sentence)
	bag = [0]*len(words)
	for w in sentence_words:
		for i, word in enumerate(words):

			# check whether the word
			# is present in the input as well
			if word == w:

				# as the list of words
				# created earlier.
				bag[i] = 1

	# return a numpy array
	return np.array(bag)

#predict_class(sentence): 
# This function will predict the class of the sentence 
# input by the user.

def predict_class(sentence):
	bow = bagw(sentence)
	res = model.predict(np.array([bow]))[0]
	ERROR_THRESHOLD = 0.25
	results = [[i, r] for i, r in enumerate(res)
			if r > ERROR_THRESHOLD]
	results.sort(key=lambda x: x[1], reverse=True)
	return_list = []
	for r in results:
		return_list.append({'intent': classes[r[0]],
							'probability': str(r[1])})
		return return_list

#get_response(intents_list, intents_json): This function will 
# print a random response from whichever 
# class the sentence/words input by the user belongs to
def get_response(intents_list, intents_json):
	tag = intents_list[0]['intent']
	list_of_intents = intents_json['intents']
	result = ""
	for i in list_of_intents:
		if i['tag'] == tag:
			
			# prints a random response
			result = random.choice(i['responses'])
			break
	return result

print("Chatbot is up!")

# Finally, we’ll initialize an infinite while loop that will prompt 
# the user for an input and print the Chatbot’s response

while True:
    message = input("")
    ints = predict_class(message)
    res = get_response(ints, intents)
    print(res)

