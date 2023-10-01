# ChatBot Using Tensorflow
Chatbots are designed to understand and respond to user queries or commands in a conversational manner, similar to how a human would engage in a conversation.

## Train the Model
We will create the file intense.json that contains all the intents, tags and word or phrases that our chatbot would responding to Then we create the training.py where we write the code for training the model.

We will use a class called WordNetLemmatizer() which will give the root words of the words that the Chatbot can recognize. For example, for hunting, hunter, hunts and hunted, the lemmatize function of the WordNetLemmatizer() class will give “hunt” because it is the root word.

## Run the Chatbot
We’re done training the model, now we need to create the main file that will make the Chatbot model work and respond to our inputs.
