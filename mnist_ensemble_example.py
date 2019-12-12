#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical as tc
import numpy as np
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


# In[ ]:


(x_train,y_train),(x_test,y_test) = mnist.load_data()


# In[ ]:


x_train,x_test=x_train/255.0,x_test/255.0


# In[ ]:


x_train_orig=x_train.copy()
y_train_orig=y_train.copy()


# In[ ]:


acc_standard = .95


# In[ ]:


test_acc=0
models=[]

old_acc = 0
# In[ ]:


def get_new_model():
    model = Sequential()
    model.add(Flatten(input_shape=(x_train[0].shape)))
    model.add(Dense(len(x_train[0].flatten()),activation='sigmoid'))
    model.add(Dense(10,activation='softmax'))
    model.compile(loss='categorical_crossentropy',optimizer='nadam')
    model.fit(x_train,tc(y_train),epochs=50,batch_size=1000)
    #print('hi')
    return(model)


# In[ ]:


def get_wrong(outs):
    temp_x = []
    temp_y = []
    for k in range(len(x_train)):
        if list(outs[k]).index(max(list(outs[k]))) != y_train[k]:
            temp_x.append(x_train[k])
            temp_y.append(y_train[k])
    return(np.array(temp_x),np.array(temp_y))


# In[ ]:
accuracies=[]

from time import sleep
test_in=[]
for k in range(60000):
    test_in.append([])
test_in = np.array(test_in)
times=[]
import time
while test_acc<acc_standard:
    start = time.time()
    models.append(get_new_model())
    temp_res = models[-1].predict(x_train_orig,batch_size=4000)
    test_in = test_in.tolist()
    for k in range(len(temp_res)):
        for n in temp_res[k]:
            test_in[k].append(n)
    test_in = np.array(test_in)
    master = Sequential()
    master.add(Flatten(input_shape=(int(len(models)*10),)))
    master.add(Dense(int(len(models)),activation='sigmoid'))
    #master.add(Dense(4,activation='sigmoid'))
    master.add(Dense(10,activation='softmax'))
    master.compile(loss='categorical_crossentropy',optimizer='nadam',metrics=['acc'])
    l = master.fit(test_in,tc(y_train_orig),epochs=50,batch_size=1000).history['acc']
    outs = master.predict(test_in,batch_size=2000)
    x_train,y_train=get_wrong(outs)
    test_acc=np.max(l)
    if len(times)>0:
        times.append(times[-1]+(time.time()-start))
    else:
        times.append(time.time()-start)
    accuracies.append(test_acc)
    if test_acc-old_acc<.1:
        break


# In[
from matplotlib import pyplot as plt
plt.figure()
plt.plot(times,accuracies)
plt.xlabel('Time (sec)')
plt.ylabel('Accuracy')
plt.show()
#plt.savefig('acc_vs_time.png',dpi=500)
