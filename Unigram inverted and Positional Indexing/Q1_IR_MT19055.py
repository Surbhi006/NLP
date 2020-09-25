#!/usr/bin/env python
# coding: utf-8

# In[217]:


import nltk
import glob
import os


# In[494]:


folder_name ='20_newsgroups'
files_path=[]
for (root,dirs,files) in os.walk(str(os.getcwd()+'/'+folder_name+'/'),topdown=False):
    for i in files:
        files_path.append(str(root)+str("/")+i)


# In[495]:


print(files_path[0])


# In[497]:


print(len(files_path))


# In[498]:


def remove_header(text):
    try:
        
        index = text.index('\n\n')
        text = text[index:]
    except:
        print("heder not present")
    return text
#remove_header(read_file)


# In[499]:


#convert to lower case
def convertToLowerCse(text):
    text = text.lower()
    #print(text)
    return text
#out_lowercase=convertToLowerCse(paths[0])


# In[500]:


#remove puntuation
import string
def punctuations(text):
    pun = string.punctuation
    for i in pun:
        text = text.replace(i," ")
    #print(text)
    return text


# In[27]:


nltk.download('stopwords')
from nltk.corpus import stopwords    
    


# In[28]:


nltk.download('punkt')
from nltk.tokenize import word_tokenize


# In[501]:


#tokenize
def tokenize(text):
    #print(type(text))
    tokens = word_tokenize(text)
    return tokens   


# In[502]:


import re
def digits_removal(text):
    digit = re.sub(r'\d', '', text)
    return digit


# In[503]:


def blank_spaces(text):
    #print(type(text))
    blank= " ".join(text).split()
    #print(type(text))
    return blank


# In[504]:


#stopwords
def stopWords(text):
    stop_words = set(stopwords.words('english'))
    lower_tokens = convertToLowerCse(text)
    tokens = tokenize(lower_tokens)
    
    new_text = []  
    for words in tokens:
        if words not in stop_words:
            new_text.append(words)
    #print(new_text)
    return new_text


# In[31]:


nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer 


# In[505]:


#lemmatization
def lemmatization(text):
    #text = ' '.join(text)
    lemmatize_text=[]
    lemmatizer = WordNetLemmatizer()
    for word in text:
        lemmatize_text.append(lemmatizer.lemmatize(word)) 
    lemmatize_text = ' '.join(lemmatize_text)
    return lemmatize_text


# In[506]:


#remove_single_words
def single_char(text):
    text = word_tokenize(text)
    #print(text)
    list_char = list (string.ascii_lowercase)
    new_list = [ele for ele in text if ele not in list_char]
    #print("222",new_list)
    return new_list


# In[507]:


def preprocessing(data):
    data = remove_header(data)
    data = convertToLowerCse(data)
    data = digits_removal(data)
    data = punctuations(data)
    data = stopWords(data)
    data = blank_spaces(data)
    data = lemmatization(data)
    data = tokenize(data)
    #data = single_char(data)
    data = list(set(data))
    return data
    


# In[508]:


doc_id=[words for words in range(1,19997+1)]
dic = dict(zip(files_path,doc_id))


# In[509]:


list_dic={}
for file_original in files_path:
    #print(file)
    with open(file_original,'r+',encoding='utf-8',errors='ignore') as file:
        
        read_file = file.read()
        
        data = preprocessing(read_file)
        
        file_name = dic.get(file_original)
        
        
        for k in data:
            if k not in list_dic:
                list_dic[k]=[file_name]
            else:
                list_dic[k].append(file_name)
#print(len(list_dic))             


# In[510]:


#frequency
for keys in list_dic:
    list_dic[keys].insert(0,len(list_dic[keys]))
    


# In[39]:


list1 = []
for i in range (0,10):
    list1.append(files_path[i])
print(list1)


# In[511]:


def gen_comm_query(query):
    query = convertToLowerCse(query)
    #print(query)
    #query = blank_spaces(query)
    #print(query)
    query = single_char(query)
    #print(query)
    #tokens = word_tokenize(str(query))
    #print(tokens)
    operators = []
    operands = []
    
    lemma = WordNetLemmatizer()
    
    for t in query:
        if t not in ['and','or','not']:
            #print("yo",str(t))
            processed_query = lemma.lemmatize(t)
            #print(processed_query)
            operands.append(str(processed_query))    
            
        else:
            operators.append(t)
    
    return operators,operands


# In[533]:


def unary_operation(query,operand):
    query = convertToLowerCse(query)
    #print(query)
    #query = blank_spaces(query)
    #print(query)
    #tokens = tokenize(str(query))
    #print(tokens)
    lemma = lemmatization(query)
    print(lemma)
    tokens = tokenize(str(query))
    print('token2',tokens)
    result = [tokens[i+1] for i,w in enumerate(tokens) if w == 'not']
    post=[[] for i in range(len(result))]
    posting_list=[[] for j in range(len(operand))]
    #print(len(posting_list))
    k=0
    for i in result:
        post1 = (list_dic[i])
        post1.pop(0)
        post2 = set(range(len(files_path)))
        post[k] = list(post2.difference(post1))
        post[k].insert(0,len(post[k]))  
        #print(operand.index(i))
        #print(post[k])
        posting_list[operand.index(i)]=post[k]
        k=k+1
    return posting_list


# In[534]:


def absence_not(not_posting,operands):
    print(type(not_posting))
    for i in range(0,len(not_posting)):
        #print(i)
        if (len(not_posting[i])==0):
            not_posting[i]=list_dic.get(operands[i])
    return not_posting


# In[531]:


operator,operands = gen_comm_query('hello and not corpus')
print(operator)


# In[538]:


post = unary_operation('hello and not corpus',operands)
print(post)


# In[539]:


new_post = absence_not(post,operands)
print(len(new_post))


# In[542]:


def and_operation(operator):
    op_list=[]
    print(operator)
    i=0
    while (i!=(len(operator))):
        present_list=[]
        while((i!=len(operator)) and operator[i]=='and'):
            present_list.append(i)
            i=i+1
        print(present_list,i)
        op_list.append(present_list)
        while((i!=len(operator)) and operator[i]=='or'):
            i=i+1
            print(i)
        print("outside",i)
        #i=i+k-1
    print(op_list)
    return op_list


# In[543]:


def calc_freq(list1):
    list2 = []
    for i in range(0,len(list1)):
        list2.append(list1[i].pop(0))
        min_index = list2.index(min(list2))
        freq_post = list1[min_index]
    return freq_post


# In[556]:


def merge_algo(list1):
    count = 0
    
    list2 = calc_freq(list1)
    
    c1 = len(list1)
    c2 = len(list2)
    temp =[]
    j=0
    k=0
    while(j<c1 and k<c2):
        if (list1[j]== list2[k]):
            count = count+1
            temp.append(list1[j])
            j= j+1
            k=k+1
        elif(list1[j]<list2[k]):
            j=j+1
            count = count+1
        else:
            k=k+1
            count = count+1
        list2 = temp          
    return list2,count  


# In[558]:


def or_operation(posting,count):
    op_list=[]
    for sublist in posting:
        #print((posting))
        for item in sublist:
            #print(type(item))
            op_list.append(item)
    size1 = len(op_list)
    size2 = len(list(set(op_list)))
    diff = size2-size1
    count = count + diff
    return op_list,count   


# In[559]:


def binary_operation(operator,operand,posting):
    and_op = and_operation(operator)
    count=0
    print(posting)
    for inner_list in and_op[:]:
        if(len(and_op)>=1):
            posting_list =[[]for i in range(len(inner_list)+1)]
            j=0
            for k in inner_list:
                posting_list[j]=posting[k]
                j=j+1
            posting_list[j]=posting[k+1]
            merge_and,count = merge_algo(posting_list)
            
            for j in range(1,len(inner_list)):
                posting.pop(inner_list[j])
            
            posting.pop(len(inner_list)-1)
            posting.insert(inner_list[0],merge_and)
            and_op.pop(0)
            for a in range(0,len(and_op)):
                for b in range(0,len(and_op[a])):
                    and_op[a][b]=and_[a][b]-len(inner_list)
    #merge_ans=[]
    merge_ans,count = or_operation(posting,count)
        
    return merge_ans,count


# In[560]:


def exe_query(query):
    operator,operands = gen_comm_query(query)
    print('operator',operator)
    print('operand',operands)
    postings = unary_operation(query,operands)
    dic_posting = absence_not(postings,operands)
    operator = list(filter(lambda i: i != 'not', operator))
    #print(dic_posting)
    documents,comparison=binary_operation(operator,operands,dic_posting)
    print("Total number of doc",documents)
    print('Total number of comparisons',comparisons)
    number_of_doc = len(documents)
    print('Number of documnts matched',number_of_doc)
    return comparison,documents,number_of_doc


# In[561]:


exe_query('not place and trees or aliens')


# In[ ]:




