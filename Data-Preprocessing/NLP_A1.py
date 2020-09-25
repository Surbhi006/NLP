# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 02:23:03 2020

@author: surbhi
"""

import string
import re
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from num2words import num2words
# In[90]:
#Count total number of sentences in a file
def countSentences(data):
    sentences = sent_tokenize(data) #tokenization
    return sentences

# In[94]:
#count total number of words in a file
def countWords(data):
  data = data.lower() #lowercase
  tokens = word_tokenize(data) #tokenization
  tokens = [i for i in tokens if not i in string.punctuation] #puntuation removal
  words=[]
  for w in tokens:
     if w.isnumeric():
         w = num2words(w) #conversion of number to words
         words.append(w)
     else:
        words.append(w)
  return words


# In[95]:
#List of all vowels Boolean function
def isVowel(ch):
    return ch == 'a' or ch == 'A' or ch == 'e' or ch == 'E' or  ch == 'i' or ch == 'I' or ch == 'o'  or ch == 'O' or ch == 'u' or ch == 'U'


# In[114]:
#list of all consonants
def isConsonant(char):
  ascii_val = ord(char)
  if(((ascii_val>=97 and ascii_val<=122) or (ascii_val>=65 and ascii_val<=90)) and isVowel(char) == False):
  #if(((char>='a' and char<='z') or (char>='A' and char<='Z')) and isVowel(char)==False):
    return True
  return False


# In[115]:
#Words strting with consonants
def countStartConsonant(words):
  cons=[]
  cons_val=0
  for w in words:
    start = w[0]
    if(isConsonant(start) and w not in stopwords.words('english')):
      cons_val+=1
      cons.append(w)
  print("\nTotal Number of Words Starting with Consonant: ",cons_val)
  print("consonant: ",cons)


# In[116]:
#Words starting with vowel
def countStartVowel(words):
  vowels=[]
  vowel_val=0
  for w in words:
    start = w[0]
    if(isVowel(start) and w not in stopwords.words('english')):
      vowel_val+=1
      vowels.append(w)
  print("\nTotal Number of Words Starting with Vowel: ",vowel_val)
  print("vowels: ",vowels)


# In[102]:
#Conversion of number to words
def number2words(word):
   if (word.isnumeric()): #check if word is numeric value then convert it into words
      w = num2words(word)
   else:
     w = word
   return w

# In[103]:
#Count sentences with input starting word
def startingWithWord(sentences,startWord):
  count = 0
  c=1
  word = number2words(startWord) #conversion num2Words
  print("\nSentences: ")
  for each_sen in sentences:
    #print(each_sen)
    curr_word = word_tokenize(each_sen) #tokenization of sentence
    if(curr_word[0].lower()==word): #checking
      count+=1
      print(c,".",each_sen,"\n") #printing sentence containing word
      c+=1
  if(count!=0): #checking if word present in start of any sentence or not
      print("\nTotal Number Of Sentences Starting With: ",startWord,": ",count) #total sentences
  else:
     print("\nNo sentence has a Starting word: ",startWord) #if word  not present


# In[104]:
#Count sentences with input ending word
def endingWithWord(sentences,endWord):
  count = 0
  size = len(endWord)
  c=1
  word = number2words(endWord) #conversion num2Words
  print("\nSentences: ")
  for each_sen in sentences:
    #print(each_sen)
    curr_word = word_tokenize(each_sen) #tokenization of sentence
    if(curr_word[-2].lower()==word): #checking
      count+=1
      print(c,".",each_sen,"\n") #printing sentence containing word
      c+=1
  if(count!=0): #checking if word present in end of any sentence or not
     print("\nTotal Number Of Sentences Ending With: ",endWord,": ",count) #total sentences
  else:
      print("\nNo sentence has a Ending word: ",endWord)#if word  not present

# In[105]:
#Count occurence of specific word
def specificWordFile(data,word):
  count=0
  word = number2words(word) #convert num2words
  tokens = word_tokenize(data) #tokenization of data
  d =[]
  for t in tokens:
      if t.isnumeric(): #if is number
          t = num2words(t)
          d.append(t)
      else:
          d.append(t)
  for w in d:
      if(word==w):
          count+=1
  #pattern = r"\b"+word+r"\b" #regex for a word with boundaries
  #tokenizer = RegexpTokenizer(pattern) 
  #line = tokenizer.tokenize(data) #searching word in file text
  if(count!=0): #checking if word present or not
      print("\nNumber of times",'(',word,')',"present in the File: ",count,"\n")
  else:
     print("\n",word,"not present in a File")


# In[106]:
#Count occurence of specific word in a sentences in a file   
def specificWordInSentenceFile(sentences,word):
  word = number2words(word) #conversion num2Words
  count=0
  c=1
#  pattern = r"\b"+word+r"\b"
  print("\nSentences: ")
  for each_sen in sentences:
        curr = word_tokenize(each_sen) #tokenization of sentence
        for _ in curr:
            w = number2words(_).lower()  #conversion num2Words
            if(w==word):
                print(c,".",each_sen)  #printing sentence containing word 
                count+=1
                c+=1
                break
  if(count!=0):  #checking if word present in any location of any sentence or not
      print("\nNumber of times",'(',word,')',"present in Sentence in the File: ",count)
  else:
      print("\nNo Sentence contains",'(',word,')',"in a File")

# In[107]:
#Count toatal number of questions present in a file.
def containsQuestion(sentences):
    regex = r"\?$" #regex for finding question mark at the end of any sentence
    count=0
    q=[]
    for ques in sentences:
        find = re.compile(regex) 
        if(find.search(ques)): #searching of question mark in a sentence
            count+=1
            q.append(ques)
    if(count!=0): #Questions present or not checking
        print("\nTotal Number of Questions",count) #printing count of questions
        print("Questions: ",q,"\n") #printing questions
    else:
        print("No Questions Present in a file")

# In[108]:
#list of all emails present
def countEmails(data):
  regex = "\S+@\S+" #regex for checking email in a file
  emails = re.findall(regex,data) #finding email pattern in a data
  print("\nEmails: ")
  c=1
  for user in emails:
    print(c,"\t",user) #print all emails
    c+=1

# In[109]:
#finding total occurences of abbrevations 
def findAbbreviations(data):
    regex = r"\b[A-Z\.]{2,}s?\b" #regex for searching U.S.A etc
    lis = re.findall(regex,data)
    print("\nAbbreviations: ")
    for item in lis:
        print(item)
    if(len(lis)!=0): #checking if present or not
        print("\nTotal Number of Abbrevations: ",len(lis))
    else:
        print("\nNo Abbreviations Present in a File.")
    
# In[112]:
#extraction of minutes and seconds from a time in a file
def extractTime(data):
  regex = "\d{2}:\d{2}:\d{2}" #digits present in this form in a file.
  time=[]
  tokens = word_tokenize(data)
  for word in tokens:
      if(re.match(regex,word)):
          minutes = word[3:5] #extracting min from time
          sec = word[-2:] #extracting seconds
          t = minutes+":"+sec 
          time.append(t)
          print(t)      
  print("Time List: ",time) #printing time list present
# In[113]:
#Removal of header in a file
def removeHeader(data):
  try:
    first_para = data.index('\n\n') #first paragraph is header in a file
    data = data[first_para+2:] 
  except: #if header not present
    print("No header")
  return data

# In[120]:

directory_name = "/home/surbhi/Assignment3_IR/20_newsgroups" #enter direction name or path with 20_newsgroup dataset present
folder = input("Enter the folder: ")
file = input("Enter the file: ")
path = directory_name+"/"+folder+"/"+file #path of a file/ change slashes according to windows and unix
try:
    file = open(path, 'r', errors = 'ignore') #reading files
    dataWithHeader = file.read() #data with header
  
    data = removeHeader(dataWithHeader)  #data without header      
    #Function calling
    words = countWords(data)
    print("\nTotal Number of words: ",len(words))
    sentences = countSentences(data)
    print("\nTotal Number Of Sentences: ",(len(sentences)))
    countStartVowel(words)
    countStartConsonant(words)
    countEmails(dataWithHeader)
    startWord = input("\nEnter the  Starting word you want to find : ")
    startingWithWord(sentences,startWord.lower())
    endWord = input("\nEnter the Ending word you want to find : ")
    endingWithWord(sentences,endWord.lower())
    word = input("\nEnter the specific word you want to search in a file or sentence : ")
    specificWordFile(data.lower(),word.lower())
    specificWordInSentenceFile(sentences,word.lower())
    containsQuestion(sentences)
    findAbbreviations(dataWithHeader)
    extractTime(dataWithHeader)
except: #if file or folder not found or any other error occured.
    print("\nException Occured")