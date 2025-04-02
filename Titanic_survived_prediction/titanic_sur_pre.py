import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# loading datasets.......

path=r'C:\Users\arunsathya\Desktop\ML\excel_data\titanicsurvival.csv'

dataset = pd.read_csv(path)

print(dataset.head(5))

# changing name into  values.........

imcome_set = set(dataset['Sex'])
dataset['Sex'] = dataset['Sex'].map({'male': 1,'female':0}).astype(int)

print(dataset.head(5))

# segregating input and output data...

X=dataset.drop('Survived',axis='columns')

Y=dataset.Survived

# checking any data missing..

X.columns[X.isna().any()]
X.Age=X.Age.fillna(X.Age.mean())

# rechecking...........
a=X.columns[X.isna().any()]


# Spliting data's  train..test..validation..

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.25,random_state=0)

# fearture scaling.........

"""sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)"""

# Algorithm..............

model=GaussianNB()
model.fit(X_train,Y_train)

# predicting...

Y_pred=model.predict(X_test)

#ACCuracy.............


print("ACCURACY  :: {0}%".format(accuracy_score(Y_test,Y_pred)*100))

Fare=float(input("Enter a FARE value ..."))
Sex=int(input("Enter a SEX FEMALE :: 0 & MALE :: 1.."))
Age=int(input("Enter a AGE ..."))
Pclass=int(input("Enter a class..."))
person=[[Pclass,Sex,Age,Fare]]
result=model.predict(person)
#result=model.predict(sc.transform(person))
print(result)
if result==1:
    print("HE/SHE will be survived..")
else:
    print("Will not be survived..")


































