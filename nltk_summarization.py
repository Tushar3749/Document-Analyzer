import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize 
from nltk.tokenize import word_tokenize, sent_tokenize
nltk.download('all')
import heapq  
from string import punctuation
from heapq import nlargest


def nltk_summarizer(raw_text):
	nltk.download("stopwords")
	stop_words = stopwords.words('english') 
	tokens = word_tokenize(raw_text)
    
	#punctuation = punctuation + '\n'

	word_frequencies = {}
	for word in tokens:
		if word.lower() not in stop_words:
			if word.lower() not in punctuation:
				if word not in word_frequencies.keys():
					word_frequencies[word] = 1
				else:
					word_frequencies[word] += 1

	maximum_frequncy = max(word_frequencies.values())

	for word in word_frequencies.keys():  
	    word_frequencies[word] = (word_frequencies[word]/maximum_frequncy)

	sent_token = sent_tokenize(raw_text)

	sentence_scores = {}
	for sent in sent_token:
		sentence = sent.split(" ")
		for word in sentence:
			if word.lower() in word_frequencies.keys():
				if sent not in sentence_scores.keys():
					sentence_scores[sent] = word_frequencies[word.lower()]
				else:
					sentence_scores[sent] += word_frequencies[word.lower()]
					
	select_length = int(len(sent_token)*0.3)

	summary = nlargest(select_length, sentence_scores, key = sentence_scores.get)

	final_summary = [word for word in summary]
	summary = ' '.join(final_summary)

	
	return summary