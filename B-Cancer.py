import pandas as pd
df=pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer/breast-cancer.data', na_values='?', header=None)

df.shape

df.isna().sum()

cols=[0,1,2,3,4,5,6,7,8,9]


from sklearn.impute import SimpleImputer
si=SimpleImputer(strategy='most_frequent')
df_im=si.fit_transform(df)

df_im=pd.DataFrame(df_im)

df.head()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
for i in cols:
  df_im[i]=le.fit_transform(df_im[i])
df_im.head()

target=df_im[0]

target.shape

data=df_im.drop(columns=[0])

data.shape

data.head()

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(data,target,test_size=0.3)


x_train.shape


x_test.shape

from sklearn.linear_model import Perceptron
p1=Perceptron(penalty='l1')
p1.fit(x_train,y_train)



from sklearn.metrics import accuracy_score

train_pred=p1.predict(x_train)

test_pred=p1.predict(x_test)


print("Training accuracy:",accuracy_score(train_pred,y_train))
#Training accuracy: 0.5

print("Testing accuracy:",accuracy_score(test_pred,y_test))
#Testing accuracy: 0.4418604651162791

from sklearn.metrics import confusion_matrix,precision_score,recall_score,f1_score


print("Confusion Matrix", confusion_matrix(test_pred,y_test))
#Confusion Matrix [[19  6]
 [42 19]]

print("Precision:", precision_score(test_pred,y_test))
#Precision: 0.76

print("Recall:", recall_score(test_pred,y_test))
#Recall: 0.3114754098360656

print("Recall:", recall_score(test_pred,y_test))
#F1-score: 0.44186046511627913




