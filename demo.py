import pandas as pd
import numpy as np
df = pd.read_csv('D:/project/Customer_Churn-Deployment-master/Churn_Modelling.csv')
print(df['Geography'].unique())
df.isnull().sum()
final_dataset = pd.get_dummies(final_dataset, drop_first=True)
X=data_m.iloc[:,3:13]
Y=data_m["Exited"]
X['Geography'].unique()
X['Gender'].unique()
geography=pd.get_dummies(X['Geography'],drop_first=True)
gender=pd.get_dummies(X['Gender'],drop_first=True)
X=X.drop(['Geography','Gender'],axis=1)
X=pd.concat([X,geography,gender],axis=1) 
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=0)
from sklearn.preprocessing import StandardScaler
sc =StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)
import tensorflow as tf
print(tf.__version__)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LeakyReLU,PReLU,ELU,ReLU
classifier=Sequential()
classifier.add(Dense(10,input_shape=(11,),activation='relu'))
classifier.add(Dense(units=10,activation='relu'))
classifier.add(Dense(units=10,activation='relu'))
classifier.add(Dense(1,activation='sigmoid'))
classifier.summary()
import tensorflow
opt=tensorflow.keras.optimizers.Adam(learning_rate=0.01)
classifier.compile(optimizer=opt,loss='binary_crossentropy',metrics=['accuracy'])
import tensorflow as tf
model_history=classifier.fit(X_train,y_train,validation_data=(X_test,y_test),batch_size=10,epochs=100)
y_pred = classifier.predict(X_test)
y_pred = (y_pred >= 0.5)
from sklearn.metrics import confusion_matrix, classification_report
cm=confusion_matrix(y_test,y_pred)
print(cm)

print("classification_report")
print(classification_report(y_pred, y_test, digits=2))
from sklearn.metrics import accuracy_score
score=accuracy_score(y_pred,y_test)
score
classifier.save("model1.h5")