from __future__ import unicode_literals
from flask import Flask,render_template,url_for,request,redirect
from flask_uploads import UploadSet,configure_uploads,ALL,DATA

from spacy_summarization import text_summarizer
from gensim.summarization import summarize
from nltk_summarization import nltk_summarizer
import time
import spacy
from urllib.request import urlopen

import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

from werkzeug.utils import secure_filename
import time
timestr = time.strftime("%Y%m%d-%H%M%S")

nlp = spacy.load('en_core_web_sm')
app = Flask(__name__)

# Web Scraping Pkg
from bs4 import BeautifulSoup
# from urllib.request import urlopen


# Sumy Pkg
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer




#------------------------------------------ DEFAULT fUNCTIONS ----------------------------------------------#

# Counting Total Words in Text
def CountWords(mytext):
	total_words = len([ token.text for token in nlp(mytext)])

	return total_words

# Fetch Text From Url
def get_text(url):
	page = urlopen(url)
	soup = BeautifulSoup(page)
	fetched_text = ' '.join(map(lambda p:p.text,soup.find_all('p')))
	return fetched_text


#------------------------------------------ DEFAULT fUNCTIONS ----------------------------------------------#




                                       
												#SUMMARIZER

#Summarizer start ------------------------------------------------------------------------------------------


@app.route('/')
def index():
	return render_template('index.html')



@app.route('/upload_summary',methods=['GET','POST'])
def upload_summary():
   start = time.time()
   if request.method == 'POST' and 'txt_data' in request.files:
        file = request.files['txt_data']
        filename = secure_filename(file.filename)
        file.save(os.path.join('static/uploadedfiles',filename))

		# Document Redaction Here
        with open(os.path.join('static/uploadedfiles',filename),'r+') as f:
            myfile = f.read()
        
        rawtext_total_words = CountWords(myfile)
        final_summary = text_summarizer(myfile)
        summary_total_words = CountWords(final_summary)
        end = time.time()
        final_time = end-start
        return render_template('index.html',ctext=myfile,final_summary=final_summary,final_time=final_time,rawtext_total_words=rawtext_total_words,summary_total_words=summary_total_words)



@app.route('/analyze',methods=['GET','POST'])
def analyze():
	start = time.time()
	if request.method == 'POST':
		rawtext = request.form['rawtext']
		rawtext_total_words = CountWords(rawtext)
		final_summary = text_summarizer(rawtext)
		summary_total_words = CountWords(final_summary)
		end = time.time()
		final_time = end-start
	return render_template('index.html',ctext=rawtext,final_summary=final_summary,final_time=final_time,rawtext_total_words=rawtext_total_words,summary_total_words=summary_total_words)

@app.route('/analyze_url',methods=['GET','POST'])
def analyze_url():
	start = time.time()
	if request.method == 'POST':
		raw_url = request.form['raw_url']
		rawtext = get_text(raw_url)
		rawtext_total_words = CountWords(rawtext)
		final_summary = text_summarizer(rawtext)
		summary_total_words = CountWords(final_summary)
		end = time.time()
		final_time = end-start
	return render_template('index.html',ctext=rawtext,final_summary=final_summary,final_time=final_time,rawtext_total_words=rawtext_total_words,summary_total_words=summary_total_words)

#Summarizer end --------------------------------------------------------------------------------------------



                                             #COMPARE SUMMARY


#Compare Summary start --------------------------------------------------------------------------------------------

@app.route('/compare_summary')
def compare_summary():
	return render_template('compare_summary.html')

@app.route('/comparer',methods=['GET','POST'])
def comparer():
   start = time.time()
   if request.method == 'POST':

        rawtext = request.form['rawtext']
        rawtext_total_words = CountWords(rawtext)
        #spacy
        final_summary_spacy = text_summarizer(rawtext)
        total_words_spacy = CountWords(final_summary_spacy)
		# Gensim Summarizer
        final_summary_gensim = summarize(rawtext)
        total_words_gensim = CountWords(final_summary_gensim)
		# NLTK
        final_summary_nltk = nltk_summarizer(rawtext)
        total_words_nltk = CountWords(final_summary_nltk)
		# Sumy
        final_summary_sumy = sumy_summary(rawtext)
        total_words_sumy =CountWords(final_summary_sumy) 

        end = time.time()
        final_time = end-start
        return render_template('compare_summary.html',ctext=rawtext,final_summary_spacy=final_summary_spacy,final_summary_gensim=final_summary_gensim,final_summary_nltk=final_summary_nltk,final_time=final_time,rawtext_total_words=rawtext_total_words, total_words_spacy= total_words_spacy, total_words_gensim= total_words_gensim,final_summary_sumy=final_summary_sumy, total_words_sumy = total_words_sumy ,total_words_nltk=total_words_nltk)


#Compare Summary end --------------------------------------------------------------------------------------------




                                              #TEXT MATCHER

#Text Matching start --------------------------------------------------------------------------------------------


@app.route('/text_matching')
def text_matching():
	return render_template('text_matching.html')
	



def check_plagiarism():
     
     
     student_files = [doc for doc in os.listdir() if doc.endswith('.txt')]
     student_notes =[open(File).read() for File in  student_files]

     vectorize = lambda Text: TfidfVectorizer().fit_transform(Text).toarray()
     similarity = lambda doc1, doc2: cosine_similarity([doc1, doc2])
     vectors = vectorize(student_notes)
     s_vectors = list(zip(student_files, vectors))
     plagiarism_results = set()
     
     

     for student_a, text_vector_a in s_vectors:
        new_vectors =s_vectors.copy()
        current_index = new_vectors.index((student_a, text_vector_a))
        del new_vectors[current_index]
        for student_b , text_vector_b in new_vectors:
            sim_score = similarity(text_vector_a, text_vector_b)[0][1]
            student_pair = sorted((student_a, student_b))
            score = (student_pair[0], student_pair[1],sim_score)
            plagiarism_results.add(score)
     return plagiarism_results







@app.route('/text_matcher',methods=['GET','POST'])
def text_matcher():

  

   if request.method == 'POST':
	   rawtxt1 = request.form['rawtext1']
	   rawtxt2 = request.form['rawtext2']
	   rawtext11 = open(r'C:/Users/Tushar Drmc\AppData/Local/Programs/Python/Python38\Scripts/DocumentAnalyzer5/file_name1.txt','w+')
	   rawtext22 = open(r'C:/Users/Tushar Drmc\AppData/Local/Programs/Python/Python38\Scripts/DocumentAnalyzer5/file_name2.txt','w+')
	   rawtext11.write(rawtxt1)
	   rawtext22.write(rawtxt2)
	   rawtext11.close()
	   rawtext22.close()
	  
	   #plagarism_checker = check_plagiarism()
	  
	  
	   plagarism_checker = set()
	   plagarism_checker.clear()

	   plagarism_checker = check_plagiarism()
	   for val in plagarism_checker:
		    per = val[2]*100
	  

   return render_template('text_matching.html',info = plagarism_checker, percent =per )



@app.route('/upload_textmatcher',methods=['GET','POST'])
def upload_textmatcher():

#    if request.method == 'POST' and 'txt_data1''textdata2' in request.files:
#         file = request.files['txt_data1']
#         filename1 = secure_filename(file.filename1)
#         file.save(os.path.join('static/text_matcher/uploadedfiles',filename1))

#         file = request.files['txt_data2']
#         filename2 = secure_filename(file.filename2)
#         file.save(os.path.join('static/text_matcher/uploadedfiles',filename2))


#         with open(os.path.join('static/text_matcher/uploadedfiles',filename1),'r+') as f:
#                myfile1 = f.read()

#         with open(os.path.join('static/text_matcher/uploadedfiles',filename2),'r+') as f:
#                myfile2 = f.read()

#         rawtext11 = open(r'C:/Users/Tushar Drmc\AppData/Local/Programs/Python/Python38\Scripts/DocumentAnalyzer5/file_name1.txt','w+')
#         rawtext22 = open(r'C:/Users/Tushar Drmc\AppData/Local/Programs/Python/Python38\Scripts/DocumentAnalyzer5/file_name2.txt','w+')
#         rawtext11.write(myfile1)
#         rawtext22.write(myfile2)
#         rawtext11.close()
#         rawtext22.close()
	  
# 	   #plagarism_checker = check_plagiarism()
        
#         plagarism_checker2 = set()
#         plagarism_checker2.clear()


#         plagarism_checker2 = check_plagiarism()
#         for val in plagarism_checker2:
#             per = val[2]*100

# info = plagarism_checker2,percent = per

   return render_template('result_text_matcher.html')

#Text Matching end --------------------------------------------------------------------------------------------



               
			                                 #SPAM DETECTION

# Spam Detection start --------------------------------------------------------------------------------------------

cv = pickle.load(open("model/vectorizer.pkl","rb"))
clf = pickle.load(open("model/model.pkl","rb"))


@app.route('/spam')
def spam():
	return render_template('spam.html')


@app.route('/predict', methods=['POST'])
def predict():
    # df = pd.read_csv("spam.csv", encoding="latin-1")
    # df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)
    # # Features and Labels
    # df['label'] = df['class'].map({'ham': 0, 'spam': 1})
    # X = df['message']
    # y = df['label']
    # # Extract Feature With CountVectorizer
    # cv = CountVectorizer()
    # X = cv.fit_transform(X)  # Fit the Data
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    # # Naive Bayes Classifier
    # clf = MultinomialNB()
    # clf.fit(X_train, y_train)
    # clf.score(X_test, y_test)
    # if request.method == 'POST':
    #     message = request.form['message']
    #     data = [message]
    #     vect = cv.transform(data).toarray()
    #     my_prediction = clf.predict(vect)
    # return render_template('spam.html', prediction=my_prediction)

    if request.method == 'POST':
     userInput = request.form.get('message')
     result = cv.transform([userInput]).toarray()
     # Predict
     pred = clf.predict(result)
     pred = int(pred[0])
     if pred == 0:
        pred=-1
     return render_template("spam.html",prediction=pred)


# Spam Detection end --------------------------------------------------------------------------------------------

# Sumy 
def sumy_summary(docx):
	parser = PlaintextParser.from_string(docx,Tokenizer("english"))
	lex_summarizer = LexRankSummarizer()
	summary = lex_summarizer(parser.document,3)
	summary_list = [str(sentence) for sentence in summary]
	result = ' '.join(summary_list)
	return result



                                           # DOCUMENT REDACTOR

# Document Redactor starts --------------------------------------------------------------------------------------

@app.route('/redactor')
def redactor():
	return render_template('redactor.html')




# Functions to Sanitize and Redact 
def sanitize_names(text):
    docx = nlp(text)
    redacted_sentences = []
    for ent in docx.ents:
        ent.merge()
    for token in docx:
        if token.ent_type_ == 'PERSON':
            redacted_sentences.append("[REDACTED NAME]")
        else:
            redacted_sentences.append(token.string)
    return "".join(redacted_sentences)

def sanitize_places(text):
    docx = nlp(text)
    redacted_sentences = []
    for ent in docx.ents:
        ent.merge()
    for token in docx:
        if token.ent_type_ == 'GPE':
            redacted_sentences.append("[REDACTED PLACE]")
        else:
            redacted_sentences.append(token.string)
    return "".join(redacted_sentences)

def sanitize_date(text):
    docx = nlp(text)
    redacted_sentences = []
    for ent in docx.ents:
        ent.merge()
    for token in docx:
        if token.ent_type_ == 'DATE':
            redacted_sentences.append("[REDACTED DATE]")
        else:
            redacted_sentences.append(token.string)
    return "".join(redacted_sentences)

def sanitize_org(text):
    docx = nlp(text)
    redacted_sentences = []
    for ent in docx.ents:
        ent.merge()
    for token in docx:
        if token.ent_type_ == 'ORG':
            redacted_sentences.append("[REDACTED]")
        else:
            redacted_sentences.append(token.string)
    return "".join(redacted_sentences)

@app.route('/sanitize',methods=['GET','POST'])
def sanitize():
    if request.method == 'POST':
        choice = request.form['taskoption']
        rawtext = request.form['rawtext']
        if choice == 'redact_names':
             result = sanitize_names(rawtext)
        elif choice == 'places':
             result = sanitize_places(rawtext)
        elif choice == 'date':
             result = sanitize_date(rawtext)
        elif choice == 'org':
             result = sanitize_org(rawtext)
        else:
             result = sanitize_names(rawtext)
        
        f1= open(r'C:/Users/Tushar Drmc\AppData/Local/Programs/Python/Python38\Scripts/DocumentAnalyzer5/static/RedactionDownload/RedactedText/RedactedData.txt','w+')
        f1.write(result)
        f1.close()

        f2= open(r'C:/Users/Tushar Drmc\AppData/Local/Programs/Python/Python38\Scripts/DocumentAnalyzer5/static/RedactionDownload/PlainText/RedactedData.txt','w+')
        f2.write(rawtext)
        f2.close()
    return render_template('redactor.html',rawtext=rawtext,result=result)


@app.route('/uploads',methods=['GET','POST'])
def uploads():
 
   if request.method == 'POST' and 'txt_data' in request.files:
        file = request.files['txt_data']
        choice = request.form['saveoption']
        filename = secure_filename(file.filename)
        file.save(os.path.join('static/uploadedfiles',filename))

		# Document Redaction Here
        with open(os.path.join('static/uploadedfiles',filename),'r+') as f:
            myfile = f.read()
            result = sanitize_names(myfile)
        if choice == 'savetotxt':
			#new_res = writetofile(result)
            return redirect(url_for('downloads'))
        elif choice == 'no_save':
            pass
        else:
            pass

   return render_template('result.html',result=result,myfile=myfile)

@app.route('/result')
def downloads():
	files = os.listdir(os.path.join('static/document_redaction/downloadfiles'))
	return render_template('result.html',files=files)

def writetofile(text):
	file_name = 'yourdocument' + timestr + '.txt'
	with open(os.path.join('static/downloadfiles',file_name),'w') as f:
		f.write(text)

# Configuration For Uploads
files = UploadSet('files',ALL)
app.config['UPLOADED_FILES_DEST'] = 'static/uploadedfiles'
configure_uploads(app,files)



# Document Redactor  end --------------------------------------------------------------------------------------------




@app.route('/about')
def about():
	return render_template('index.html')




if __name__ == '__main__':
	app.run(debug=True)