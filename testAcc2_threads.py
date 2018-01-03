import tensorflow as tf
import pickle
import numpy as np
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
from bottle import Bottle, ServerAdapter, route, run, request, template
import re
import math
import itertools
import sys
import html as htmla
import threading, time, random
import queue as Queue

train_x,train_y,test_x,test_y = pickle.load(open("jobs_set_10000_0.pickle","rb"))
qfeatures=Queue.Queue(900)
qlabels=Queue.Queue(900)
finishedflag=0
outputarray=[]
errorcount=0

#####################################################################
#####################################################################
lemmatizer = WordNetLemmatizer()

n_nodes_hl1 = 1000
n_nodes_hl2 = 500
n_nodes_hl3 = 1000

n_classes = 64029
with open('lexicon_jt_64029.pickle','rb') as f:
    print('label')
    labellexicon = pickle.load(f)
    print(len(labellexicon))
with open('lexicon_jd_22694.pickle','rb') as f:
    print('features')
    featurelexicon = pickle.load(f)
    print(len(featurelexicon))


x = tf.placeholder('float',shape=(None, 22694))
y = tf.placeholder('float',shape=(None, 64029))

hidden_1_layer = {'f_fum':n_nodes_hl1,
                  'weight':tf.Variable(tf.random_normal([22694, n_nodes_hl1])),
                  'bias':tf.Variable(tf.random_normal([n_nodes_hl1]))}

hidden_2_layer = {'f_fum':n_nodes_hl2,
                  'weight':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                  'bias':tf.Variable(tf.random_normal([n_nodes_hl2]))}
hidden_3_layer = {'f_fum':n_nodes_hl3,
                  'weight':tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
                  'bias':tf.Variable(tf.random_normal([n_nodes_hl2]))}

output_layer = {'f_fum':None,
                'weight':tf.Variable(tf.random_normal([n_nodes_hl2, n_classes])),
                'bias':tf.Variable(tf.random_normal([n_classes])),}

#####################################################################
#                       DEFINE NETWORK HERE                         #
#####################################################################
def neural_network_model(data):
    l1 = tf.add(tf.matmul(data,hidden_1_layer['weight']), hidden_1_layer['bias'])
    l1 = tf.nn.relu(l1)
    l2 = tf.add(tf.matmul(l1,hidden_2_layer['weight']), hidden_2_layer['bias'])
    l2 = tf.nn.tanh(l2)#relu(l2)
    l3 = tf.add(tf.matmul(l1,hidden_2_layer['weight']), hidden_2_layer['bias'])
    l3 = tf.nn.tanh(l3)
    output = tf.matmul(l3,output_layer['weight']) + output_layer['bias']
    #print(output)
    return output


saver = tf.train.import_meta_graph('./modelMiddle.ckpt.meta')
#
#
#
class Producer:
    def __init__(self,test_x,train_y,featurelexicon,labellexicon):
        self.test_x=test_x
        self.test_y=test_y
        self.nextTime=0
        self.i=0
        self.featurelexicon=featurelexicon
        self.labellexicon=labellexicon
        self.end=False
        
    def __del__(self):
        print ("deleted")
        
    def run(self):
        global qfeatures,qlabels,errorcount
        while (self.i < len(self.test_x)):
            if (self.i/100).is_integer():
                print('inserting: '+str(self.i))
            batch_x = []
            batch_y = []
            
            if(self.nextTime<time.clock()):
                jobDescription = self.test_x[self.i]
                                            
                features = np.zeros(len(self.featurelexicon))

                jobDescription =jobDescription.strip().split()
                
                for word in jobDescription:
                    
                    if word.lower() in self.featurelexicon:
                       
                        index_value = self.featurelexicon.index(word.lower())
                        features[index_value] += 1
                line_x = list(features)
                #----------------------------------------------------
                #                        y
                #----------------------------------------------------

                jobtitle = self.test_y[self.i]
                            
                label = np.zeros(len(self.labellexicon))                        
        
                fline=jobtitle.strip()                

                
                if fline.lower() in self.labellexicon:
                                               
                    index_value = self.labellexicon.index(fline.lower())
                    
                    label[index_value] += 1
                    line_y = list(label)
                
                
                qfeatures.put(line_x)
                qlabels.put(line_y)
                self.nextTime+=(random.random())/1000
                self.i+=1
           
            fline=''
            jobtitle=''
            current_words=[]


class Consumer:
    def __init__(self,maximum,sess,prediction):
        self.nextTime=1
        self.max=(maximum)
        self.i=0
        self.sess=sess
        self.prediction=prediction
        #self.optimizer=optimizer
       # self.cost=cost

    def __del__(self):
        print ("deleted")
        
    def run(self):
        
        global qfeatures,finishedflag,qlabels,errorcount
        batch_x=[]
        batch_y = []
        
        while (self.i<(self.max-(errorcount))):
           if(self.nextTime<time.clock() and not qfeatures.empty()):
                batch_x.append(qfeatures.get())
                batch_y.append(qlabels.get())
                features = np.array(list(batch_x))

               
                result=self.prediction.eval(session = self.sess,feed_dict={x: np.array(batch_x)})
                result= np.array(result)
                outputarray.append(str((labellexicon[int(np.argmax(result))])))
               
                batch_x = []
                batch_y = []
                self.nextTime+=(random.random()*2)/1000
                self.i+=1
                        
        finishedflag=1
               
#####################################################################
#                       USE NETWORK HERE                            #
#####################################################################
def use_neural_network():
    global finishedflag
    global train_x,train_y,test_x,test_y
    global outputarray
    global errorcount
    hit=0
    count=0
    
    prediction = neural_network_model(x)
   

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        p=Producer(test_x,test_y,featurelexicon,labellexicon)
        c=Consumer(len(test_x),sess,prediction)
        pt=threading.Thread(target=p.run,args=())
        ct=threading.Thread(target=c.run,args=())
        pt.start()
        time.sleep(2)   
        ct.start()

        while (finishedflag==0):
            time.sleep(20)
        print('ending')
        print(finishedflag)

        for i in range(1000):
            print(test_y[i])
            print(outputarray[i])
            
        if test_y[i] == outputarray[i]:
            
            hit+=1
    total = len(test_y)

    print(hit)
    print(total)

use_neural_network()   




