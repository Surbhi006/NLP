#!/usr/bin/env python
# coding: utf-8

# In[32]:


import nltk
import os
import string
import re
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


# In[44]:


files_path=[]
basepath = '20_newsgroups/comp.graphics'
for fname in os.listdir(basepath):
    #print(fname)
    path = os.path.join(basepath, fname)
    files_path.append(path)
#print(len(files_path))


# In[45]:


basepath = '20_newsgroups/rec.motorcycles'
for fname in os.listdir(basepath):
    #print(fname)
    path = os.path.join(basepath, fname)
    files_path.append(path)
#print(len(files_path))


# In[46]:


print(len(files_path))


# In[4]:


def remove_header(text):
    try:
        index = text.index('\n\n')
        text = text[index:]
    except:
        print("heder not present")
    return text


# In[75]:


def convertToLowerCse(text):
    return text.lower()


# In[6]:


def punctuations(text):
    pun = string.punctuation
    for i in pun:
        text = text.replace(i," ")
    return text


# In[7]:


def digits_removal(text):
    digit = re.sub(r'\d', '', text)
    return digit


# In[8]:


def blank_spaces(text):
    blank= " ".join(text).split()
    return blank


# In[53]:


def stopWords(text):
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text)
    
    new_text = []  
    for words in tokens:
        if words not in stop_words:
            new_text.append(words)
    return new_text


# In[10]:


def lemmatization(text):
    lemmatize_text=[]
    lemmatizer = WordNetLemmatizer()
    for word in text:
        lemmatize_text.append(lemmatizer.lemmatize(word)) 
    lemmatize_text = ' '.join(lemmatize_text)
    return lemmatize_text


# In[79]:


def single_char(text):
    text = word_tokenize(text)
    list_char = list (string.ascii_lowercase)
    new_list = [ele for ele in text if ele not in list_char]
    return (new_list)


# In[153]:


def preprocessing(data):
    data = remove_header(data)
    data = convertToLowerCse(data)
    data = digits_removal(data)
    data = punctuations(data)
    data = stopWords(data)
    data = blank_spaces(data)
    data = lemmatization(data)
    data = single_char(str(data))
    data = list((data))
    #print(type(data))
    #print(data)
    return data
    


# In[189]:


doc_id=[words for words in range(1,2000+1)]
dic = dict(zip(files_path,doc_id))
print(dic)


# In[173]:


dictionary_inner = {}
#docid = 1
outer_dictionary = {}
for f in files_path:
    with open(f,'r+',encoding='utf-8',errors='ignore') as file:
        
        read_content = file.read()
        
        data = preprocessing(read_content)
        #print(type(data))
        file_name = dic.get(f)
        #print(file_name)
        pos = 1
        for i in data:
            
            if i not in outer_dictionary:
                
                outer_dictionary[i]={}
            
                outer_dictionary[i][file_name]=[]
                outer_dictionary[i][file_name].append(pos)
                #print("hey")
            
            else:
                
                if file_name not in outer_dictionary[i].keys():
                    #print("bye")
                    outer_dictionary[i][file_name]=[]
                    outer_dictionary[i][file_name].append(pos)
                else:
                    
                    outer_dictionary[i][file_name].append(pos)                   
            
            pos = pos+1   
print((outer_dictionary))


# In[221]:


def merge_algo(list1,list2):
    c1 = len(list1)
    c2 = len(list2)
    temp =[]
    j=0
    k=0
    while(j<c1 and k<c2):
        if list1[j]== list2[k]:
            temp.append(list1[j])
            j=j+1
            k=k+1
        elif list1[j]<list2[k]:
            j=j+1
        else:
            k=k+1
    return temp


# In[222]:


def exe_query(list1,list2):
    c1 = len(list1)
    c2 = len(list2)
    temp =[]
    j=0
    k=0
    while(j<c1 and k<c2):
        if (list1[j]== list2[k]):
            j=j+1
            k=k+1
        elif(list1[j]+1 == list2[k]):
            return True
        elif(list1[j]<list2[k]):
            j=j+1
        else:
            k=k+1
    return False  


# In[225]:


def retrieval_doc(query):
    query = convertToLowerCse(query)
    query = word_tokenize(query)
    #print("Query",query)
    if(len(query)>1):
        doc_list=[]
        #print("tokens",query)
        token1=[]
        
        token1=query[0]
        print(token1)
        print(len(outer_dictionary))
        dic = outer_dictionary.get(token1)
        print((dic))
        
        for i in range(1,len(query)):
            token2 = query[i]
            print(len(token2))
            dic1 = outer_dictionary.get(token2)
            print((dic1))
            new_list=[]
            for doc in dic.keys():
                if doc in dic1.keys():
                    ans = exe_query(dic[doc],dic[doc])
                    #print(ans)
                    if ans == True:
                        new_list.append(doc)
            dic = dic1
            token1 = token2
            doc_list.append(new_list)
            post_list1=doc_list[0]
            c=0
            for j in range(1,len(doc_list)):
                post_list2=doc_list[j]
                post_list1=merge_algo(post_list1,post_list2)
            return post_list1
        if len(query) == 1:
            return outer_dictionary[querry[0]].keys()


# In[224]:


#phrase_query = convertToLowerCse(phrase_query)
#phrase_query = word_tokenize(phrase_query)
new_list = retrieval_doc('projection line projected')
print(len(new_list))


# In[ ]:




