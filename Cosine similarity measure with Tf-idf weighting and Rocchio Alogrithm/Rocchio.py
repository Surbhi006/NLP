#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import os
from nltk.tokenize import word_tokenize
from num2words import num2words
import nltk
import string
import unicodedata
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import pickle
import math
import numpy as np
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE


# In[ ]:


folder_name ='Assignment3_IR/20_newsgroups'
files_path=[]
total_doc = 0
folders = ['comp.graphics','sci.med','talk.politics.misc','rec.sport.hockey','sci.space']
for folder in folders:
    for files in (os.listdir(folder_name+'/'+folder)):
        i=str(folder_name+'/'+folder+'/'+files)
        total_doc+=1
        files_path.append(i)


# In[ ]:


print(len(files_path),total_doc)


# In[ ]:


doc_id=[words for words in range(1,len(files_path)+1)]
dic = dict(zip(files_path,doc_id))


# In[ ]:


directory = 'Assignment3_IR/20_newsgroups'
mapping = {}
itr = 1
for i in range(len(folders)):
    for files in os.listdir(directory+'/'+folders[i]):
        mapping[itr] = folders[i]+" "+files
        itr+=1


# In[ ]:


def removeHeader(text):
    paragraphs = text.split('\n\n')
    metadata_removed_text = ""
    text_list = []
    for i in range(1 , len(paragraphs)):
        metadata_removed_text = metadata_removed_text + paragraphs[i]
    return metadata_removed_text


# In[ ]:


def preprocess_data(text):
    token_list=[]
    ps = nltk.PorterStemmer() 
    stop_words = set(stopwords.words('english'))
    text = removeHeader(text)
    text = text.lower()
    text = word_tokenize(text)
    punc = string.punctuation
    tokens = [i for i in text if not i in (stop_words and punc)]
    for word in tokens:
        w=word.translate(str.maketrans('','',string.punctuation))
        if len(w)>1 and w!= '' and w not in stop_words:
            w=ps.stem(w)
            if w.isnumeric():
                w = num2words(w)
                token_list.append(w)
            else:
                token_list.append(w)
    return token_list


# In[ ]:


def preprocess_query(text):
    token_list=[]
    ps = nltk.PorterStemmer() 
    stop_words = set(stopwords.words('english'))
    #text = removeHeader(text)
    text = text.lower()
    text = word_tokenize(text)
    punc = string.punctuation
    tokens = [i for i in text if not i in (stop_words and punc)]
    for word in tokens:
        w=word.translate(str.maketrans('','',string.punctuation))
        if len(w)>1 and w!= '' and w not in stop_words:
            w=ps.stem(w)
            if w.isnumeric():
                w = num2words(w)
                token_list.append(w)
            else:
                token_list.append(w)
    return token_list


# In[ ]:


def loadpkl():
    picfile = open('Assignment4_Q1.pkl','rb')
    picfile=pickle.load(picfile)
    pos=picfile[0]
    return pos
text_dic = loadpkl()


# In[ ]:


text_dic = loadpkl()


# In[ ]:


print(len(text_dic))


# In[ ]:


tf_dict={}
for term in text_dic:
    tf_dict[term]={}
    for lst in text_dic[term]:
        tf_dict[term][lst[0]]=lst[1]


# In[ ]:


idf={}
for term in text_dic:
    idf[term] = math.log(total_doc/len(text_dic[term]), 10)


# In[ ]:


tf_idf = {}
for term in tf_dict:
    tf_dict_val = 0
    tf_idf_inner = {}
    for doc in tf_dict[term]:
        tf_dict_val = tf_dict[term][doc] * idf[term]
        tf_idf_inner[doc] = tf_dict_val
    tf_idf[term] = tf_idf_inner 
print(len(tf_idf))   


# In[ ]:


query = input("Enter query: ")
k=input("enter k: ")
k=int(k)
pre_query=preprocess_query(query)
pre_query


# In[ ]:


tf = {}
for term in pre_query:
    if term not in tf:
        tf[term] = pre_query.count(term)/len(pre_query)
query_tf_idf={}
for term in pre_query:
    if term not in text_dic:
        print("out of vocab",term)
    else:
        if term not in query_tf_idf:
            query_tf_idf[term]=tf[term]*idf[term]


# In[ ]:


def union(a,b):
    a=set(a)
    b=set(b)
    c=a|b
    c=list(c)
    #print(c)
    return c


# In[ ]:


doc_dic={}
for term in text_dic:
    doc_id = []
    for lst in text_dic[term]:
        doc_id.append(lst[0])
    doc_dic[term] = doc_id


# In[ ]:


a=[]
for term in pre_query:
    b=[]
    for lst in text_dic[term]:
        b.append(lst[0])
    #print(len(b))
    a = union(a,b)


# In[ ]:


def queryVector():
    query_vector = []
    for term in text_dic:
        if term in pre_query:
            query_vector.append(query_tf_idf[term])
        else:
            query_vector.append(0)
    return query_vector


# In[ ]:


def loadpkl():
    picfile = open('A4_Q1.pkl','rb')
    picfile=pickle.load(picfile)
    pos=picfile[0]
    return pos
docs_dic = loadpkl()


# In[ ]:


len(docs_dic)


# In[ ]:


from itertools import islice

def take(n, iterable):
    "Return first n items of the iterable as a list"
    return list(islice(iterable, n))
n_items = take(10,docs_dic.items())
print(n_items)


# In[ ]:


for docs in docs_dic:
        docs_dic[docs] = np.asarray(docs_dic[docs])


# In[ ]:


def cosine_sim(Q):
    ranked_doc={}
    for docs in a:
        vec1 = np.dot(docs_dic[docs],Q)
        vec1 = np.dot(vec1,vec1)
        temp = np.sum(vec1)
        value_DQ = math.sqrt(temp)
        document = np.dot(docs_dic[docs],docs_dic[docs])
        temp1 = np.sum(document)
        d = math.sqrt(temp1)
        query = np.dot(Q,Q)
        temp2 = np.sum(query)
        q = math.sqrt(temp2)
        try:
            ranked_doc[docs] = value_DQ/d*q
        except:
            ranked_doc[docs] = 0
    sorted_dictionary = sorted(ranked_doc.items(), reverse = True ,key = lambda x:x[1])
    sorted_dictionary =  [i for i in sorted_dictionary] 
    return sorted_dictionary


# In[ ]:


def mapp_doc(ranked_doc):
    sorted_dictionary = ranked_doc
    sorted_dictionary= sorted_dictionary[:(k)]
    ret_doc=[]
    orgDoc = []
    for i in sorted_dictionary:
        ret_doc.append(i[0])
    for doc in ret_doc:
        #print(doc)
        orgDoc.append(mapping[doc])
    return orgDoc,ret_doc 


# In[ ]:


query_vector = queryVector()
query_vector = np.asarray(query_vector)


# In[ ]:


def user_feedback(retrived_doc):
    enter_rel = input("Enter number of relevant docs: ").split(',')
    rel_doc = [retrived_doc[int(i)] for i in enter_rel]
    non_rel = list(set(retrived_doc)-set(rel_doc))
    return rel_doc,non_rel


# In[ ]:


def calc_centroid(docs,const):
    vec = np.asarray(docs_dic[docs[0]])
    #print(vec)
    for d_Id in range(1,len(docs)):
        vec+=docs_dic[docs[d_Id]]
    vec = (const/len(docs)) * vec
    return vec


# In[ ]:


def modified_query(alpha,beta,gamma,query_vector,rel_docs,non_rel_docs): 
    query_vector = np.multiply(alpha,query_vector)
    D_r = calc_centroid(rel_docs,beta)
    D_nr = calc_centroid(non_rel_docs,gamma)
    sum1 = np.add(query_vector,D_r)
    q_m = np.subtract(sum1,D_nr)
    return q_m


# In[ ]:


def compute_MAP(relevant_doc,retrieved_doc):
    t=1
    r=0
    AP=0
    for i in range(1,len(retrieved_doc)):
        if retrieved_doc[i] in relevant_doc:
            r+=1
            p=r/t
            AP+=p
        t+=1
    MAP = AP/len(relevant_doc)
    return MAP


# In[ ]:


def plot_TSNE(relevant_doc,non_relevant_doc,vector_q):
    X = []
    labels = []
    
    for d_Id in relevant_doc:
        X.append(docs_dic[d_Id])
        labels.append(0)
    for d_Id in non_relevant_doc:
        X.append(docs_dic[d_Id])
        labels.append(1)
    X.append(query_vector)
    labels.append(2) 
    X = np.asarray(X)
    X_embedded = TSNE(n_components = 2, verbose=0, random_state=0).fit_transform(X)
    X_embedded = np.asarray(X_embedded)
    print("Xshape: ",X.shape)
    x_axis = X_embedded[:,0]
    y_axis = X_embedded[:,1]
    #plt.scatter(x_axis,y_axis,c=labels, s=60, alpha =0.8)
    colormap = np.array(['tab:red', 'tab:blue', 'tab:green'])
    groups = np.array(["R", "N-R", "Query"])  
    plt.scatter(x_axis, y_axis,c=colormap[labels], label = groups)
    plt.title("Rocchio")
    plt.show()    


# In[ ]:


def top_retrieved(ret_doc,rel_doc,orgDoc):
    print("Top docs: ")
    for i in range(len(ret_doc)):
        if ret_doc[i] in rel_doc:
            print(i,ret_doc[i],orgDoc[i],'*')
        else:
            print(i,ret_doc[i],orgDoc[i])


# In[ ]:


Map=[]
rank = cosine_sim(query_vector)
orgDoc,ret_doc=mapp_doc(rank)
for i in range(len(ret_doc)):
    print(i,ret_doc[i],orgDoc[i])
rel_doc,non_rel = user_feedback(ret_doc)
precision = []
recall = []
total_relevant = 1000
total_retrieved = 0
relevant_retrieved = 0 
for i in range(len(ret_doc)):
    if(ret_doc[i] in rel_doc):
        relevant_retrieved+=1
    total_retrieved+=1
    p = relevant_retrieved/total_retrieved
    r = relevant_retrieved/total_relevant
    precision.append(p)
    recall.append(r)
MAP = compute_MAP(rel_doc,ret_doc)
print('MAP: ',MAP)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.plot(recall, precision)
plt.show()
plot_TSNE(rel_doc,non_rel,query_vector)


# In[ ]:


flag = 0

for iteration in range(1,5):
    print("iteration",iteration)
    query_vector = modified_query(1,0.75,0.25,query_vector,rel_doc,non_rel)
    for i in range(len(query_vector)):
        if(query_vector[i]<0):
            query_vector[i]=0
    retrived_docs = cosine_sim(query_vector)
    orgDoc,ret_doc=mapp_doc(retrived_docs) 
    top_retrieved(ret_doc,rel_doc,orgDoc)
    rel_doc,non_rel = user_feedback(ret_doc)
    print(rel_doc,non_rel)
    precision_m = []
    recall_m = []
    total_relevant_m = 1000
    total_retrieved_m = 0
    relevant_retrieved_m = 0 
    for i in range(len(ret_doc)):
        if(ret_doc[i] in rel_doc):
            relevant_retrieved_m+=1
        total_retrieved_m+=1
        p_m = relevant_retrieved_m/total_retrieved_m
        r_m = relevant_retrieved_m/total_relevant_m
        precision_m.append(p_m)
        recall_m.append(r_m)
    MAP_m = compute_MAP(rel_doc,ret_doc)
    Map.append(MAP_m)
    print('MAP: ',MAP_m,Map)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.plot(recall_m, precision_m)
    plt.show()
    plot_TSNE(rel_doc,non_rel,query_vector)


# In[ ]:


plt.plot(Map)
plt.show()


# In[ ]:


sum = 0
for i in Map:
    sum+=i
sum = sum/len(Map)
print("MAP",sum)

