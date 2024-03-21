# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. import the reuqired and print the present data.
2. find the null and duplicate values.
3. using logistic find the predicted values of accuracy ,confusion matrices.
4. display the results

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: C.shrenidhi
RegisterNumber:  212223040196
*/
import pandas as pd
data=pd.read_csv("/content/Placement_Data.csv")
data.head()
data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)
data1.head()
data.isnull()
data1.duplicated().sum()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"] )
data1["status"]=le.fit_transform(data1["status"])
data1
x=data1.iloc[:,:-1]
x
y=data1["status"]
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred
from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy
from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,y_pred)
confusion
from sklearn.metrics import classification_report
classification_report=classification_report(y_test,y_pred)
print(classification_report)
lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])

```

## Output:
#PLACEMENT DATA
<img width="844" alt="image" src="https://github.com/AkilaMohan/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/155261096/7df61bd2-c296-4302-a798-e42c778a2156">

#Null data
<img width="749" alt="image" src="https://github.com/AkilaMohan/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/155261096/4f446cb4-8716-4f91-930d-46f379c3e200">

#X DATA
<img width="626" alt="image" src="https://github.com/AkilaMohan/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/155261096/1ef9bb6c-0abf-45ea-8326-e0e9e451c6b0">

#Y DATA
<img width="254" alt="image" src="https://github.com/AkilaMohan/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/155261096/3d0362ce-1141-46f7-a963-be524372c8df">

#Y Predicted
<img width="876" alt="image" src="https://github.com/AkilaMohan/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/155261096/e824db6e-8b36-4652-96ff-ed685c66e711">

#Accuracy



<img width="325" alt="image" src="https://github.com/AkilaMohan/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/155261096/61f05093-52e1-47fb-b64c-eeaaa5fe8958">

#Classifiaction report



<img width="344" alt="image" src="https://github.com/AkilaMohan/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/155261096/60caa92a-b22a-4e1f-a546-6112b70457b5">















## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
