import pandas as pd
import numpy as np
from time import time

train = pd.read_csv('vowel.train')
train = train.sample(frac=1).reset_index(drop=True)
test = pd.read_csv('vowel.test')
X_train = train.iloc[:,2:]
y_train = train['y']
X_test = test.iloc[:,2:]
y_test = test['y']



print('train data shape: ',X_train.shape)
print('test data shape: ',X_test.shape)



from sklearn.ensemble import GradientBoostingClassifier

''' Training a Gradient Boosting Classifier with following parameters:

Number of trees: 300
Learning rate: 0.1
Max depth of each tree: 3

Since Tree Classifier doesn't need scaled features, so we feed the input directly to algorithm without preprocessing
'''
gbc = GradientBoostingClassifier(n_estimators=300,random_state=18)
t = time()
gbc.fit(X_train,y_train)
print('Training time: %.2fs' % (time()-t))
print('Train accuracy score: %.2f%%' % (100*gbc.score(X_train,y_train)))
print('Test accuracy score: %.2f%%' % (100*gbc.score(X_test,y_test)))

# write hisotry accuracy path  on training and test data after each epoch to csv
r = []
for i,j in enumerate(zip(gbc.staged_predict(X_train),gbc.staged_predict(X_test))):
    r.append([i+1,np.log10(i+1),(j[0]==y_train).mean(),(j[1]==y_test).mean()])
pd.DataFrame(r,columns=['epoch','log10 epoch','train acc','test acc']).to_csv('gbc.csv',index=False)


from sklearn.ensemble import RandomForestClassifier

''' Training a Random Forest Classifier with following parameters:

number of trees: 200
max depth of each tree: 9
minimum samples required for each leaf: 5

We feed the input directly to algorithm without preprocessing, since Tree Classifier doesn't need scaled features
'''
rfc = RandomForestClassifier(n_estimators=200,max_depth=9,min_samples_leaf=5,random_state=18)
t = time()
rfc.fit(X_train,y_train)
print('Training time: %.2fs' % (time()-t))
print('Train accuracy score: %.2f%%' % (100*rfc.score(X_train,y_train)))
print('Test accuracy score: %.2f%%' % (100*rfc.score(X_test,y_test)))



from sklearn.neighbors import KNeighborsClassifier

''' Training a K Nearest Neighbor Classifier with default parameters:

Number of neighbor: 5
Distance metrics: Euclidian distance
'''

KNN = KNeighborsClassifier()
KNN.fit(X_train,y_train)
t = time()
print('Training time: %.2fs' % (time()-t))
print('Train accuracy score: %.2f%%' % (100*KNN.score(X_train,y_train)))
print('Test accuracy score: %.2f%%' % (100*KNN.score(X_test,y_test)))



from sklearn.svm import SVC


''' Training a Support Vector Classifier with following parameters:

Kernel: Radial basis function
Decision function: One VS One
Gamma (Kernel coefficient): 0.3
'''

svc = SVC(gamma=0.3,decision_function_shape='ovo')
t = time()
svc.fit(X_train,y_train)
print('Training time: %.2fs' % (time()-t))
print('Train accuracy score: %.2f%%' % (100*svc.score(X_train,y_train)))
print('Test accuracy score: %.2f%%' % (100*svc.score(X_test,y_test)))



import keras
from keras import optimizers, Input
from keras.models import Model
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import BatchNormalization



'''Scale the input into the range from 0 to 1 which works better with neural network;
Max of X_train is not more than 6, and min of X_train is not less than -6;
Dummify target variable required;
We augmente data to make it 16 times larger by adding normally distribution random noice to each input feature with zero 
mean, 0.005 variance
'''
X_train_nn = (X_train+6)/12
X_test_nn = (X_test+6)/12
Y_train = pd.get_dummies(y_train)
Y_test = pd.get_dummies(y_test)
for i in range(4):
    X_train_nn = pd.concat([X_train_nn,X_train_nn+np.random.normal(0,0.005,X_train_nn.shape)]).reset_index(drop=True)
    Y_train = pd.concat([Y_train,Y_train]).reset_index(drop=True)

# Define Model Structure
model=Sequential()
model.add(Dense(64,activation='sigmoid',input_shape=(X_train.shape[1],)))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(Dense(Y_train.shape[1],activation='softmax'))
optimizer = optimizers.Adam(lr=0.01, decay=1e-6, beta_1=0.9, beta_2=0.999, amsgrad=False)
model.compile(loss='categorical_crossentropy',optimizer=optimizer,metrics=['accuracy'])
t = time()
hist = model.fit(X_train_nn,Y_train.values,verbose=0,batch_size=32,epochs=300,validation_data=(X_test_nn.values,Y_test.values))
# This time including scoring validation data after each epoch
print('Training time: %.2fs' % (time()-t))
print('Accuracy score: %.2f%%' % (100*((np.argmax(model.predict(X_test_nn.values),1)+1)==y_test).mean()))

# write hisotry accuracy path  on training and test data after each epoch to csv
r = []
for i,j in enumerate(zip(hist.history['acc'],hist.history['val_acc'])):
    r.append([i+1,np.log10(i+1),j[0],j[1]])
pd.DataFrame(r,columns=['epoch','log10 epoch','train acc','test acc']).to_csv('nn_64.csv',index=False)



'''Scale the input into the range from 0 to 1 which works better with neural network;
Max of X_train is not more than 6, and min of X_train is not less than -6;
Dummify target variable required;
We augmente data to make it 16 times larger by adding normally distribution random noice to each input feature with zero 
mean, 0.005 variance
'''
X_train_nn = (X_train+6)/12
X_test_nn = (X_test+6)/12
Y_train = pd.get_dummies(y_train)
Y_test = pd.get_dummies(y_test)
for i in range(4):
    X_train_nn = pd.concat([X_train_nn,X_train_nn+np.random.normal(0,0.005,X_train_nn.shape)]).reset_index(drop=True)
    Y_train = pd.concat([Y_train,Y_train]).reset_index(drop=True)

# Define Model Structure
model=Sequential()
model.add(Dense(128,activation='sigmoid',input_shape=(X_train.shape[1],)))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(Dense(Y_train.shape[1],activation='softmax'))
optimizer = optimizers.Adam(lr=0.01, decay=1e-6, beta_1=0.9, beta_2=0.999, amsgrad=False)
model.compile(loss='categorical_crossentropy',optimizer=optimizer,metrics=['accuracy'])
t = time()
hist = model.fit(X_train_nn,Y_train.values,verbose=0,batch_size=32,epochs=300,validation_data=(X_test_nn.values,Y_test.values))
# This time including scoring validation data after each epoch
print('Training time: %.2fs' % (time()-t))
print('Accuracy score: %.2f%%' % (100*((np.argmax(model.predict(X_test_nn.values),1)+1)==y_test).mean()))

# write hisotry accuracy path  on training and test data after each epoch to csv
r = []
for i,j in enumerate(zip(hist.history['acc'],hist.history['val_acc'])):
    r.append([i+1,np.log10(i+1),j[0],j[1]])
pd.DataFrame(r,columns=['epoch','log10 epoch','train acc','test acc']).to_csv('nn_128.csv',index=False)



'''Scale the input into the range from 0 to 1 which works better with neural network;
Max of X_train is not more than 6, and min of X_train is not less than -6;
Dummify target variable required;
We augmente data to make it 16 times larger by adding normally distribution random noice to each input feature with zero 
mean, 0.005 variance
'''
X_train_nn = (X_train+6)/12
X_test_nn = (X_test+6)/12
Y_train = pd.get_dummies(y_train)
Y_test = pd.get_dummies(y_test)
for i in range(4):
    X_train_nn = pd.concat([X_train_nn,X_train_nn+np.random.normal(0,0.005,X_train_nn.shape)]).reset_index(drop=True)
    Y_train = pd.concat([Y_train,Y_train]).reset_index(drop=True)

# Define Model Structure
model=Sequential()
model.add(Dense(256,activation='sigmoid',input_shape=(X_train.shape[1],)))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(Dense(Y_train.shape[1],activation='softmax'))
optimizer = optimizers.Adam(lr=0.01, decay=1e-6, beta_1=0.9, beta_2=0.999, amsgrad=False)
model.compile(loss='categorical_crossentropy',optimizer=optimizer,metrics=['accuracy'])
t = time()
hist = model.fit(X_train_nn,Y_train.values,verbose=0,batch_size=32,epochs=300,validation_data=(X_test_nn.values,Y_test.values))
# This time including scoring validation data after each epoch
print('Training time: %.2fs' % (time()-t))
print('Accuracy score: %.2f%%' % (100*((np.argmax(model.predict(X_test_nn.values),1)+1)==y_test).mean()))

# write hisotry accuracy path  on training and test data after each epoch to csv
r = []
for i,j in enumerate(zip(hist.history['acc'],hist.history['val_acc'])):
    r.append([i+1,np.log10(i+1),j[0],j[1]])
pd.DataFrame(r,columns=['epoch','log10 epoch','train acc','test acc']).to_csv('nn_256.csv',index=False)

