from subprocess import call
import tkinter as tk
import tkinter as tk
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
#import tkinter as tk
from PIL import Image, ImageTk
from tkinter import ttk
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score,roc_curve

from sklearn.metrics import confusion_matrix, accuracy_score

root = tk.Tk()
root.title("train")

w, h = root.winfo_screenwidth(), root.winfo_screenheight()
root.geometry("%dx%d+0+0" % (w, h))
# ++++++++++++++++++++++++++++++++++++++++++++

image2 = Image.open('sample1.jpg')

image2 = image2.resize((w, h), Image.ANTIALIAS)

background_image = ImageTk.PhotoImage(image2)


background_label = tk.Label(root, image=background_image)
background_label.image = background_image
background_label.place(x=0, y=0)  # , relwidth=1, relheight=1)


label_l2 = tk.Label(root, text="___CROP PREDICTION___",font=("times", 30, 'bold','italic'),
                    background="green", fg="white", width=70, height=2)
label_l2.place(x=0, y=0)


# _+++++++++++++++++++++++++++++++++++++++++++++++++++++++
data = pd.read_csv("D:/Project/Crop Yield Prediction/Crop.csv")



data = data.dropna()

le = LabelEncoder()



def Data_Preprocessing():
    data = pd.read_csv("D:/Project/Crop Yield Prediction/Crop.csv")
    data.head()

    data = data.dropna()

    """One Hot Encoding"""

    le = LabelEncoder()

    data['Season'] = le.fit_transform(data['Season'])
    
    data['Area'] = le.fit_transform(data['Area'])
    
    data['Rainfall'] = le.fit_transform(data['Rainfall'])
    
    data['avg_temp'] = le.fit_transform(data['avg_temp'])

    data['PH'] = le.fit_transform(data['PH'])
    
    data['soil_type'] = le.fit_transform(data['soil_type'])
 
    
    """Feature Selection => Manual"""
    x = data.drop(['Crop'], axis=1)
    data = data.dropna()

    print(type(x))
    y = data['Crop']
    print(type(y))
    x.shape

    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20)

    

    load = tk.Label(root, font=("Tempus Sans ITC", 15, "bold"), width=50, height=2, background="green",
                    foreground="white", text="Data Loaded=>Splitted into 80% for Training & 20% for Testing")
    load.place(x=200, y=80)


def SVM():
    data = pd.read_csv("D:/Project/Crop Yield Prediction/Crop.csv")
    data.head()

    data = data.dropna()

    """One Hot Encoding"""

    le = LabelEncoder()

    data['Season'] = le.fit_transform(data['Season'])
    
    data['Area'] = le.fit_transform(data['Area'])
 
    data['Rainfall'] = le.fit_transform(data['Rainfall'])
    
    data['avg_temp'] = le.fit_transform(data['avg_temp'])

    data['PH'] = le.fit_transform(data['PH'])
    
    data['soil_type'] = le.fit_transform(data['soil_type'])
 
    
    """Feature Selection => Manual"""
    x = data.drop(['Crop'], axis=1)
    data = data.dropna()

    print(type(x))
    y = data['Crop']
    print(type(y))
    x.shape

    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20,random_state=10)

    from sklearn.svm import SVC
    svcclassifier = SVC(kernel='linear',random_state=10)
    svcclassifier.fit(x_train, y_train)

    y_pred = svcclassifier.predict(x_test)
    print(y_pred)

    
    print("=" * 40)
    print("==========")
    print("Classification Report : ",(classification_report(y_test, y_pred)))
    print("Accuracy : ",accuracy_score(y_test,y_pred)*100)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))
    ACC = (accuracy_score(y_test, y_pred) * 100)
    repo = (classification_report(y_test, y_pred))
   
    label4 = tk.Label(root,text =str(repo),width=45,height=10,bg='khaki',fg='black',font=("Tempus Sanc ITC",14))
    label4.place(x=205,y=200)
    
    label5 = tk.Label(root,text ="Accracy : "+str(ACC)+"%\nModel saved as SVM_MODEL.joblib",width=45,height=3,bg='khaki',fg='black',font=("Tempus Sanc ITC",14))
    label5.place(x=205,y=420)
    from joblib import dump
    dump (svcclassifier,"SVM_MODEL.joblib")
    print("Model saved as SVM_MODEL.joblib")

def DT():
    data = pd.read_csv("D:/Project/Crop Yield Prediction/Crop.csv")
    data.head()

    data = data.dropna()

    """One Hot Encoding"""

    le = LabelEncoder()

    data['Season'] = le.fit_transform(data['Season'])
    
    data['Area'] = le.fit_transform(data['Area'])
 
    data['Rainfall'] = le.fit_transform(data['Rainfall'])
    
    data['avg_temp'] = le.fit_transform(data['avg_temp'])

    data['PH'] = le.fit_transform(data['PH'])
    
    data['soil_type'] = le.fit_transform(data['soil_type'])
 
    
    """Feature Selection => Manual"""
    x = data.drop(['Crop'], axis=1)
    data = data.dropna()

    print(type(x))
    y = data['Crop']
    print(type(y))
    x.shape

    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20,random_state=10)

    from sklearn.svm import SVC
    from sklearn.tree import DecisionTreeClassifier
    clf = DecisionTreeClassifier()
    clf.fit(x_train, y_train)

    y_pred = clf.predict(x_test)
    print(y_pred)

    
    print("=" * 40)
    print("==========")
    print("Classification Report : ",(classification_report(y_test, y_pred)))
    print("Accuracy : ",accuracy_score(y_test,y_pred)*100)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))
    ACC = (accuracy_score(y_test, y_pred) * 100)
    repo = (classification_report(y_test, y_pred))
    
    
    label4 = tk.Label(root,text =str(repo),width=45,height=10,bg='khaki',fg='black',font=("Tempus Sanc ITC",14))
    label4.place(x=205,y=200)
    
    label5 = tk.Label(root,text ="Accracy : "+str(ACC)+"%\nModel saved as DT_MODEL.joblib",width=45,height=3,bg='khaki',fg='black',font=("Tempus Sanc ITC",14))
    label5.place(x=205,y=420)
    from joblib import dump
    dump (clf,"DT_MODEL.joblib")
    print("Model saved as DT_MODEL.joblib")


def NB():
    data = pd.read_csv("D:/Project/Crop Yield Prediction/Crop.csv")
    data.head()

    data = data.dropna()

    """One Hot Encoding"""

    le = LabelEncoder()
    
    data['Season'] = le.fit_transform(data['Season'])
    
    data['Area'] = le.fit_transform(data['Area'])

    data['Rainfall'] = le.fit_transform(data['Rainfall'])
    
    data['avg_temp'] = le.fit_transform(data['avg_temp'])

    data['PH'] = le.fit_transform(data['PH'])
    
    data['soil_type'] = le.fit_transform(data['soil_type'])
 
    
    """Feature Selection => Manual"""
    x = data.drop(['Crop'], axis=1)
    data = data.dropna()

    print(type(x))
    y = data['Crop']
    print(type(y))
    x.shape

    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20,random_state=6)

   # from sklearn.svm import SVC
    from sklearn.naive_bayes import GaussianNB  
    clf = GaussianNB() 
    clf.fit(x_train, y_train)

    y_pred = clf.predict(x_test)
    print(y_pred)
    cm = confusion_matrix(y_test,y_pred)
    cm
    accuracy = (cm[0][0]+cm[1][1])/(cm[0][1] + cm[1][0] +cm[0][0] +cm[1][1])
    
    print("=" * 40)
    print("==========")
    print("Classification Report : ",(classification_report(y_test, y_pred)))
    print("Accuracy : ",accuracy_score(y_test,y_pred)*100)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))
    ACC = (accuracy_score(y_test, y_pred) * 100)
    repo = (classification_report(y_test, y_pred))
    
    
    label4 = tk.Label(root,text =str(repo),width=45,height=10,bg='khaki',fg='black',font=("Tempus Sanc ITC",14))
    label4.place(x=205,y=200)
    
    label5 = tk.Label(root,text ="Accracy : "+str(ACC)+"%\nModel saved as NB_MODEL.joblib",width=45,height=3,bg='khaki',fg='black',font=("Tempus Sanc ITC",14))
    label5.place(x=205,y=420)
    from joblib import dump
    dump (clf,"NB_MODEL.joblib")
    print("Model saved as NB_MODEL.joblib")



def RF():
    
    data = pd.read_csv("D:/Project/Crop Yield Prediction/Crop.csv")
    data.head()

    data = data.dropna()

    """One Hot Encoding"""

    le = LabelEncoder()
  
    data['Season'] = le.fit_transform(data['Season'])
    
    data['Area'] = le.fit_transform(data['Area'])
  
    data['Rainfall'] = le.fit_transform(data['Rainfall'])
    
    data['avg_temp'] = le.fit_transform(data['avg_temp'])

    data['PH'] = le.fit_transform(data['PH'])
    
    data['soil_type'] = le.fit_transform(data['soil_type'])
 
    
    """Feature Selection => Manual"""
    x = data.drop(['Crop'], axis=1)
    data = data.dropna()

    print(type(x))
    y = data['Crop']
    print(type(y))
    x.shape

    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20,random_state=10)

   

    from sklearn.ensemble import RandomForestClassifier as RF
    classifier = RF(n_estimators=14, criterion='entropy', random_state=64)
    classifier.fit(x_train,y_train)
    y_pred = classifier.predict(x_test)

    
    print("=" * 40)
    print("==========")
    print("Classification Report : ",(classification_report(y_test, y_pred)))
    print("Accuracy : ",accuracy_score(y_test,y_pred)*100)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))
    ACC = (accuracy_score(y_test, y_pred) * 100)
    repo = (classification_report(y_test, y_pred))
    
    
    label4 = tk.Label(root,text =str(repo),width=45,height=10,bg='khaki',fg='black',font=("Tempus Sanc ITC",14))
    label4.place(x=205,y=200)
    
    label5 = tk.Label(root,text ="Accracy : "+str(ACC)+"%\nModel saved as RF_MODEL.joblib",width=45,height=3,bg='khaki',fg='black',font=("Tempus Sanc ITC",14))
    label5.place(x=205,y=420)
    from joblib import dump
    dump (classifier,"RF_MODEL.joblib")
    print("Model saved as RF_MODEL.joblib")

    

def ADA():
    
    data = pd.read_csv("D:/Project/Crop Yield Prediction/Crop.csv")
    data.head()

    data = data.dropna()

    """One Hot Encoding"""

    le = LabelEncoder()
  
    data['Season'] = le.fit_transform(data['Season'])
    
    data['Area'] = le.fit_transform(data['Area'])
 
    data['Rainfall'] = le.fit_transform(data['Rainfall'])
    
    data['avg_temp'] = le.fit_transform(data['avg_temp'])

    data['PH'] = le.fit_transform(data['PH'])
    
    data['soil_type'] = le.fit_transform(data['soil_type'])
 
    
    """Feature Selection => Manual"""
    x = data.drop(['Crop'], axis=1)
    data = data.dropna()

    print(type(x))
    y = data['Crop']
    print(type(y))
    x.shape

    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20,random_state=10)

    from sklearn.ensemble import AdaBoostClassifier
    clf = AdaBoostClassifier(n_estimators=50,learning_rate=1)
    clf.fit(x_train, y_train)

    y_pred = clf.predict(x_test)
    print(y_pred)
    
    print("=" * 40)
    print("==========")
    print("Classification Report : ",(classification_report(y_test, y_pred)))
    print("Accuracy : ",accuracy_score(y_test,y_pred)*100)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))
    ACC = (accuracy_score(y_test, y_pred) * 100)
    repo = (classification_report(y_test, y_pred))
    
    
    label4 = tk.Label(root,text =str(repo),width=45,height=10,bg='khaki',fg='black',font=("Tempus Sanc ITC",14))
    label4.place(x=205,y=200)
    
    label5 = tk.Label(root,text ="Accracy : "+str(ACC)+"%\nModel saved as ADA_MODEL.joblib",width=45,height=3,bg='khaki',fg='black',font=("Tempus Sanc ITC",14))
    label5.place(x=205,y=420)
    from joblib import dump
    dump (clf,"ADA_MODEL.joblib")
    print("Model saved as ADA_MODEL.joblib")
    


def call_file():
   from subprocess import call
   call(['python','Check_Prediction.py'])





def window():
    root.destroy()

button2 = tk.Button(root, foreground="white", background="black", font=("Tempus Sans ITC", 14, "bold"),
                    text="Data_Preprocessing", command=Data_Preprocessing, width=15, height=2)
button2.place(x=5, y=120)


button3 = tk.Button(root, foreground="white", background="black", font=("Tempus Sans ITC", 14, "bold"),
                    text="SVM Training", command=SVM, width=15, height=2)
button3.place(x=5, y=220)


button3 = tk.Button(root, foreground="white", background="black", font=("Tempus Sans ITC", 14, "bold"),
                    text="DT Training", command=DT, width=15, height=2)
button3.place(x=5, y=320)


button3 = tk.Button(root, foreground="white", background="black", font=("Tempus Sans ITC", 14, "bold"),
                    text="NB Training", command=NB, width=15, height=2)
button3.place(x=5, y=420)

button4 = tk.Button(root, foreground="white", background="black", font=("Tempus Sans ITC", 14, "bold"),
                    text="RF Training", command=RF, width=15, height=2)
button4.place(x=5, y=520)

button5 = tk.Button(root, foreground="white", background="black", font=("Tempus Sans ITC", 14, "bold"),
                    text="ADA Training", command=ADA, width=15, height=2)
button5.place(x=5, y=620)


button6 = tk.Button(root, foreground="white", background="black", font=("Tempus Sans ITC", 14, "bold"),
                    text="Crop Yield prediction", command=call_file, width=15, height=2)
button6.place(x=5, y=720)
exit = tk.Button(root, text="Exit", command=window, width=15, height=2, font=('times', 15, ' bold '),bg="red",fg="white")
exit.place(x=5, y=820)

root.mainloop()

'''+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++'''