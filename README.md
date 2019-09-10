# PhilbotUSRepository
Corpora Text
A chatbot is a piece of software that conducts a conversation via auditory or textual methods.[1] Such programs are often designed to convincingly simulate how a human would behave as a conversational partner, although as of 2019, they are far short of being able to pass the Turing test.[2] Chatbots are typically used in dialog systems for various practical purposes including customer service or information acquisition. Some chatbots use sophisticated natural language processing systems, but many simpler ones scan for keywords within the input, then pull a reply with the most matching keywords, or the most similar wording pattern, from a database.
The term "ChatterBot" was originally coined by Michael Mauldin (creator of the first Verbot, Julia) in 1994 to describe these conversational programs.[3] Today, most chatbots are accessed via virtual assistants such as Google Assistant and Amazon Alexa, via messaging apps such as Facebook Messenger or WeChat, or via individual organizations' apps and websites.[4][5] Chatbots can be classified into usage categories such as conversational commerce (e-commerce via chat), analytics, communication, customer support, design, developer tools, education, entertainment, finance, food, games, health, HR, marketing, news, personal, productivity, shopping, social, sports, travel and utilities.[6]
Beyond chatbots, Conversational AI refers to the use of messaging apps, speech-based assistants and chatbots to automate communication and create personalized customer experiences at scale.[7]

Chatbot Code

#Meet Robo: your friend
	
	#import necessary libraries
	import io
	import random
	import string # to process standard python strings
	import warnings
	import numpy as np
	from sklearn.feature_extraction.text import TfidfVectorizer
	from sklearn.metrics.pairwise import cosine_similarity
	import warnings
	warnings.filterwarnings('ignore')
	
	import nltk
	from nltk.stem import WordNetLemmatizer
	nltk.download('popular', quiet=True) # for downloading packages
	
	# uncomment the following only the first time
	#nltk.download('punkt') # first-time use only
	#nltk.download('wordnet') # first-time use only
	
	
	#Reading in the corpus
	with open('chatbot.txt','r', encoding='utf8', errors ='ignore') as fin:
	    raw = fin.read().lower()
	
	#TOkenisation
	sent_tokens = nltk.sent_tokenize(raw)# converts to list of sentences 
	word_tokens = nltk.word_tokenize(raw)# converts to list of words
	
	# Preprocessing
	lemmer = WordNetLemmatizer()
	def LemTokens(tokens):
	    return [lemmer.lemmatize(token) for token in tokens]
	remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)
	def LemNormalize(text):
	    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))
	
	
	# Keyword Matching
	GREETING_INPUTS = ("hello", "hi", "greetings", "sup", "what's up","hey",)
	GREETING_RESPONSES = ["hi", "hey", "*nods*", "hi there", "hello", "I am glad! You are talking to me"]
	
	def greeting(sentence):
	    """If user's input is a greeting, return a greeting response"""
	    for word in sentence.split():
	        if word.lower() in GREETING_INPUTS:
	            return random.choice(GREETING_RESPONSES)
	
	
	# Generating response
	def response(user_response):
	    robo_response=''
	    sent_tokens.append(user_response)
	    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
	    tfidf = TfidfVec.fit_transform(sent_tokens)
	    vals = cosine_similarity(tfidf[-1], tfidf)
	    idx=vals.argsort()[0][-2]
	    flat = vals.flatten()
	    flat.sort()
	    req_tfidf = flat[-2]
	    if(req_tfidf==0):
	        robo_response=robo_response+"I am sorry! I don't understand you"
	        return robo_response
	    else:
	        robo_response = robo_response+sent_tokens[idx]
	        return robo_response
	
	
	flag=True
	print("ROBO: My name is Robo. I will answer your queries about Chatbots. If you want to exit, type Bye!")
	while(flag==True):
	    user_response = input()
	    user_response=user_response.lower()
	    if(user_response!='bye'):
	        if(user_response=='thanks' or user_response=='thank you' ):
	            flag=False
	            print("ROBO: You are welcome..")
	        else:
	            if(greeting(user_response)!=None):
	                print("ROBO: "+greeting(user_response))
	            else:
	                print("ROBO: ",end="")
	                print(response(user_response))
	                sent_tokens.remove(user_response)
	    else:
	        flag=False
	        print("ROBO: Bye! take care..")    
