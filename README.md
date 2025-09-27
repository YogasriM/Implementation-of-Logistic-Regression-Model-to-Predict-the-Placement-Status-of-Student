# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required packages and print the present data.

2.Print the placement data and salary data.

3.Find the null and duplicate values.

4.Using logistic regression find the predicted values of accuracy , confusion matrices.

5.Display the results. 

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: 
RegisterNumber:  
*/

import pandas as pd
data=pd.read_csv("Placement_Data.csv")
data.head()

data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)#Browses the specified row or column
data1.head()

data1.isnull().sum()

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
classification_report1 = classification_report(y_test,y_pred)
print(classification_report1)
lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
```

## Output:
<img width="1259" height="255" alt="Screenshot 2025-09-27 094307" src="https://github.com/user-attachments/assets/e4e002b8-3c27-481c-b800-c0f2cf063d00" />

<img width="1407" height="269" alt="Screenshot 2025-09-27 094255" src="https://github.com/user-attachments/assets/7b79619f-147e-4b66-a9d0-f12e6fdb7c86" />

<img width="223" height="57" alt="Screenshot 2025-09-27 094322" src="https://github.com/user-attachments/assets/295c3419-a0dd-4485-b60e-ab0948f5395a" />

<img width="296" height="611" alt="Screenshot 2025-09-27 094315" src="https://github.com/user-attachments/assets/15101086-9feb-4a21-9f51-726733f890ed" />

<img width="1154" height="530" alt="Screenshot 2025-09-27 094332" src="https://github.com/user-attachments/assets/fbad74fe-6296-44f1-9b6e-c09b75eb2dec" />

<img width="275" height="565" alt="Screenshot 2025-09-27 094340" src="https://github.com/user-attachments/assets/33572076-a2d4-4020-bb61-db05afd293dd" />

<img width="747" height="69" alt="Screenshot 2025-09-27 094349" src="https://github.com/user-attachments/assets/5b986ac3-5a7c-4952-af81-48c1ae2ec945" />

<img width="268" height="40" alt="Screenshot 2025-09-27 094355" src="https://github.com/user-attachments/assets/31b72254-c174-479d-af11-0f3a9347e1db" />

<img width="297" height="61" alt="Screenshot 2025-09-27 094402" src="https://github.com/user-attachments/assets/73fcb8f2-47a6-41e5-85c1-ea67529deee1" />

<img width="674" height="233" alt="Screenshot 2025-09-27 094409" src="https://github.com/user-attachments/assets/dced0ffc-016d-45b2-ab6e-310c7ec76d23" />

<img width="379" height="43" alt="Screenshot 2025-09-27 094233" src="https://github.com/user-attachments/assets/4905b500-5f20-4d98-892e-4e5172974cf6" />

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
