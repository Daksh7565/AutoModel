import pandas as pd
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.svm import SVC,SVR
from sklearn.linear_model import LogisticRegression,LinearRegression
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.metrics import  accuracy_score,mean_squared_error
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier,XGBRegressor
import pickle as pk
import numpy as np
from imblearn.over_sampling import SMOTE
import keras
from tensorflow import keras
from tensorflow.keras.models import Model
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import img_to_array
import matplotlib.pyplot as plt
import random
import cv2
import os
from tensorflow.keras import optimizers
from PIL import Image
DATASET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Dataset")
import warnings
warnings.filterwarnings('ignore')
import splitfolders
from keras import activations
from keras import models, layers
def info(path):
    q=DATASET_DIR+path
    a=pd.read_csv(q)
    r,c=a.shape[0],a.shape[1]
    b=a.columns
    attributes=b
    return [r,c,attributes]
def check(path,output):
    p = DATASET_DIR
    o = p + path
    a = pd.read_csv(o)
    n=a[output]
    c=d=e=f=0
    q={}
    for i in n:
        if i not in q:
            q[i]=1
        else:
            q[i]+=1
    if max(q.values())<20:
        c+=1
    else:
        if max(q.values())- min(q.values())>min(q.values()):
            f+=1
    v=a.columns
    g={}
    for i in v:
        if i not in g:
            g[i]=a[i].isnull().sum()

    if a.isnull().sum().sum()>0:
        d+=1
    for i in a.columns:
        if a[i].dtype=="object":
            e+=1
            break
    return c,d,e,f,q,g
def null(path):
    q = DATASET_DIR + path
    a = pd.read_csv(q)
    b = a.shape
    w = a.columns
    p=a.isnull().sum()
    e=a.columns
    k=t=0
    o={}
    for i in e:
        if a[i].isnull().sum()>0:
            if k<a[i].isnull().sum():
                k=a[i].isnull().sum()
            t+=a[i].isnull().sum()
        if i not in o:
            o[i]=a[i].isnull().sum()
    y=0
    if k>((b[0]*50)/100) or t>((b[0]*50)/100):
        y+=1
    m = {}
    if y==0:
        a.dropna(inplace=True)
    else:
        for i in o:
            if o[i]>0:
                x=a[i].dtype
                if x=="int64":
                    a[i].fillna(a[i].mean(),inplace=True)
                    if i not in m:
                        m[i]=a[i].isnull().sum()
                elif x=="float64":
                    a[i].fillna(a[i].median(),inplace=True)
                    if i not in m:
                        m[i]=a[i].isnull().sum()
                elif x=="object":
                    a[i].fillna(a[i].mode(),inplace=True)
                    if i not in m:
                        m[i]=a[i].isnull().sum()
    n = a.isnull().sum()


    return a
def preproces(path,output):
    q=DATASET_DIR+path
    a=pd.read_csv(q)
    b=a.shape
    w=a.columns
    hh=a[output].value_counts()
    ii=0
    kk=0
    for i in hh:
        ii+=1
    if ii<10: 
        kk+=1
        #print("the number of row and columns are",b[0],b[1])
        a=null(path)
        c=a.info()
        u=[]
        for i in w:
            if a[i].dtype=="int64" or a[i].dtype=='float64':
                #print("the ",i,"have numercial value")
                pass
            else:
                u.append(i)
                #print("the ",i,"is the having object value")
        Lab=LabelEncoder()
        #print("the object element are",u)
        for i in u:
            a[i]=Lab.fit_transform(a[i])
        for i in w:
            if a[i].dtype=="int64" or a[i].dtype=='float64':
                #print("the ",i,"have numercial value")
                pass
            else:
                #print("the ", i, "is the having object value")
                pass
        #print(c)
        #if b[0]<500:
            #print("dataset is so small")
        s=a.drop(columns=output)
        d=a[output]
        tr=SMOTE(sampling_strategy="minority")
        bb,cc=tr.fit_resample(s,d)
        
        if min(d)!=0:
            cc=Lab.fit_transform(cc)
            return bb,cc,a,w,b,kk
        else:
            return bb,cc,a,w,b,kk
    else:
        #print("the number of row and columns are",b[0],b[1])
        a=null(path)
        c=a.info()
        u=[]
        for i in w:
            if a[i].dtype=="int64" or a[i].dtype=='float64':
                #print("the ",i,"have numercial value")
                pass
            else:
                u.append(i)
                #print("the ",i,"is the having object value")
        Lab=LabelEncoder()
        #print("the object element are",u)
        for i in u:
            a[i]=Lab.fit_transform(a[i])
        for i in w:
            if a[i].dtype=="int64" or a[i].dtype=='float64':
                #print("the ",i,"have numercial value")
                pass
            else:
                #print("the ", i, "is the having object value")
                pass
        #print(c)
        #if b[0]<500:
            #print("dataset is so small")
        s=a.drop(columns=output)
        d=a[output]
        
        return s,d,a,w,b,kk
def train(path,output):
    s,d,a,w,b,hh=preproces(path,output)
    if hh==1:
        z,x,c,v= train_test_split(s,d,test_size=0.1)
        # print("Input is ",s)
        # print("the output is ",d)
        # print("the split train data is ",z,c)
        models = [ LogisticRegression(max_iter=10000), SVC(), RandomForestClassifier(random_state=0),XGBClassifier()]
        p=[]
        tu={}
        def compare():
            c=0
            f=0
            for model in models:
                cv_score = cross_val_score(model, s, d, cv=5)
                mean = sum(cv_score) / len(cv_score)
                mean = mean * 100
                mean = round(mean, 2)
                if c<mean:
                    c=mean
                    f=models.index(model)
                tu[model]=mean
                #print('cross validation accuracies for this ', model, "=", cv_score)
                #print("accuracy of the ", model, "=", mean, "%")
                #print("<---------------------------------------------------------->")
            p.append(models[f])
            #print("the best model for the Given Dataset is ",models[f],"with the accuracy of ",c)
        compare()
        f=p[0]
        if f.fit(z,c):
            #print("model is ready")
            pass
        m=f.predict(x)
        n=accuracy_score(v,m)
        model_filename = "best_model.sav"
        with open(model_filename, "wb") as file:
            pk.dump(f, file)
        filename = "modified_dataset.csv"
        a.to_csv(filename, index=False)
        #print("the final accuracy at test data is ",n*100)
        return [w,b[0],b[1],p,n*100,tu,model_filename,filename,f,a]
    else:
        z,x,c,v= train_test_split(s,d,test_size=0.1)
        # print("Input is ",s)
        # print("the output is ",d)
        # print("the split train data is ",z,c)
        models = [ LinearRegression(), SVR(), RandomForestRegressor(random_state=0),XGBRegressor()]
        p=[]
        tu={}
        def compare():
            c=0
            f=0
            for model in models:
                cv_score = cross_val_score(model, s, d, cv=5)
                mean = sum(cv_score) / len(cv_score)
                mean = mean * 100
                mean = round(mean, 2)
                if c<mean:
                    c=mean
                    f=models.index(model)
                tu[model]=mean
                #print('cross validation accuracies for this ', model, "=", cv_score)
                #print("accuracy of the ", model, "=", mean, "%")
                #print("<---------------------------------------------------------->")
            p.append(models[f])
            #print("the best model for the Given Dataset is ",models[f],"with the accuracy of ",c)
        compare()
        f=p[0]
        if f.fit(z,c):
            #print("model is ready")
            pass
        m=f.predict(x)
        n=mean_squared_error(v,m)
        model_filename = "best_model.sav"
        with open(model_filename, "wb") as file:
            pk.dump(f, file)
        filename = "modified_dataset.csv"
        a=pd.merge(s,d,left_index=True,right_index=True)
        a.to_csv(filename, index=False)
        return [w,b[0],b[1],p,n/100,tu,model_filename,filename,f,a]
def op(path,output):
    uu = train(path,output)
    with open(uu[6], "wb") as file:
        pk.dump(uu[8], file)
    filename = "modified_dataset.csv"
    uu[9].to_csv(filename, index=False)
    #print("the final accuracy at test data is ",n*100)
    return uu    
def NLP(path,i,o,token):
    a=pd.read_csv(path)
    a.dropna(inplace=True)
    s=a[i]
    d=a[o]
    if d.dtype=="int64":
        pass
    else:
        l=LabelEncoder()
        o=l.fit_transform(d)
        d=o
    q,w,e,r= train_test_split(s,d,test_size=0.1)
    if token=="TFID":
        from sklearn.feature_extraction.text import TfidfVectorizer
        k=TfidfVectorizer( min_df =1 ,stop_words='english', lowercase=True)
        j = k.fit_transform(q)
        h=k.transform(w)
        g=e.astype('int')
        tt=r.astype('int')
    else:
        from sklearn.feature_extraction.text import CountVectorizer
        k=TfidfVectorizer( min_df =1 ,stop_words='english', lowercase=True)
        j = k.fit_transform(q)
        h=k.transform(w)
        g=e.astype('int')
        tt=r.astype('int')
    models = [ LogisticRegression(max_iter=10000), SVC(), RandomForestClassifier(random_state=0),XGBClassifier()]
    p=[]
    tu={}
    def compare():
        c=0
        f=0
        for model in models:
            cv_score = cross_val_score(model, j, g, cv=5)
            mean = sum(cv_score) / len(cv_score)
            mean = mean * 100
            mean = round(mean, 2)
            if c<mean:
                c=mean
                f=models.index(model)
            tu[model]=mean
            #print('cross validation accuracies for this ', model, "=", cv_score)
            #print("accuracy of the ", model, "=", mean, "%")
            #print("<---------------------------------------------------------->")
        p.append(models[f])
        #print("the best model for the Given Dataset is ",models[f],"with the accuracy of ",c)
    compare()
    f=p[0]
    if f.fit(j,g):
        #print("model is ready")
        pass
    m=f.predict(h)
    n=accuracy_score(tt,m)
    #print("the final accuracy at test data is ",n*100)
    model_filename = "best_model.sav"
    with open(model_filename, "wb") as file:
        pk.dump(f, file)

    return [models,p,n*100,tu,model_filename]
def image(path,q,m):
    a="C:\Image"+q
    output_folder=a
    ratio=(0.7, 0.2, 0.1)
    if m=="CNN":
        target_size=(180,180)
    else:
        target_size=(224,224)
    batch_size=16
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Perform dataset split
    splitfolders.ratio(path, output=output_folder, seed=42, ratio=ratio)
    folder=[ratio]
    # Define paths for train, validation, and test sets
    train_dir = os.path.join(output_folder, "train")
    val_dir = os.path.join(output_folder, "val")
    test_dir = os.path.join(output_folder, "test")
    train_datagen = ImageDataGenerator(
    rescale= (1./255),
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

    val_datagen = ImageDataGenerator(rescale=(1./255))
    
    
    
    
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size,
        batch_size=16,
        color_mode='rgb',
        class_mode='categorical')
    
    
    val_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size,
        batch_size=16,
        color_mode='rgb',
        class_mode='categorical')
    filename=os.listdir(train_dir)
    folder=[ratio,filename,target_size,batch_size]
    if m=="CNN":
        model = models.Sequential()
        model.add(layers.Conv2D(filters=32, kernel_size=(3,3), activation='relu',input_shape=(180,180,3)))
        model.add(layers.MaxPool2D(pool_size=(2,2)))
        model.add(layers.Conv2D(filters=64,kernel_size=(3,3),activation= 'relu'))
        model.add(layers.MaxPool2D(pool_size=(2,2)))
        model.add(layers.Conv2D(filters=128,kernel_size=(3,3),activation= 'relu'))
        model.add(layers.MaxPool2D(pool_size=(2,2)))
        model.add(layers.Conv2D(filters=256,kernel_size=(3,3),activation= 'relu'))
        model.add(layers.MaxPool2D(pool_size=(2,2)))
        model.add(layers.Dropout(rate=0.5))
        model.add(layers.Flatten())
        model.add(layers.Dense(len(filename), activation ='softmax'))
    elif m=="Resnet50":
        base_model = tf.keras.applications.ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        base_model.trainable = False
        x=base_model.output
        x = base_model.output
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        output = layers.Dense(len(filename), activation='softmax')(x)
        model = Model(inputs=base_model.input, outputs=output)
    elif m=="Mobilenet":
        base_model = tf.keras.applications.MobileNet(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        base_model.trainable = False
        x=base_model.output
        x = base_model.output
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        output = layers.Dense(len(filename), activation='softmax')(x)
        model = Model(inputs=base_model.input, outputs=output)
    
    model.compile(
    loss='categorical_crossentropy',
    optimizer="adam",
    metrics=['accuracy']
    )
    loss='categorical_crossentropy'
    optimizer="adam"
    metrics='accuracy'
    x = "model.sav"
    with open(x, "wb") as file:
        pk.dump(model, file)
    
    history = model.fit(train_generator,
    epochs=10,
    validation_data = val_generator
    )
    model_detail=[x,[loss,optimizers,metrics],10,history]
    return folder, model_detail


def ttrain(path,output,p,o,u,y,t):
    q=DATASET_DIR+path
    a=pd.read_csv(q)
    b=a.shape
    w=a.columns
    q={}
    for i in w:
        if a[i].isnull().sum()>0:
            if i not in q:
                q[i]=a[i].isnull().sum()
    zzz=["Dropping the row", "Applying imputation on the data"]
    xxx=["Yes","No"]
    ccc=["Recommend the Best model","select the Specific model"]
    vvv={"Logistic Regression":0,"RandomForest":2,"SVC":1,"XGBoost":3}
    
    if p==zzz[0]:
        a.dropna(inplace=True)
    elif p==zzz[1]:
        for i in q:
            if a[i].dtype=="int64":
                a[i].fillna(a[i].mean(),inplace=True)
            elif a[i].dtype=="float64":
                a[i].fillna(a[i].median(),inplace=True)
            elif a[i].dtype=="object":
                a[i].fillna(a[i].mode(),inplace=True)
    e=[]
    l=LabelEncoder()
    for i in w:
        if a[i].dtype=="object":
            e.append(i)
    for i in e:
        a[i]=l.fit_transform(a[i])
    if min(a[output])!=0:
        a[output]=l.fit_transform(a[output])
    s=a.drop(columns=output)
    d=a[output]
    if o==xxx[0]:
        r=SMOTE()
        f,g=r.fit_resample(s,d)
    else:
        f=s
        g=d
    qq,ww,ee,rr=train_test_split(f,g,test_size=u,random_state=42)
    if y==ccc[0]:
        models = [LogisticRegression(max_iter=10000), SVC(), RandomForestClassifier(random_state=0), XGBClassifier()]
        p = []
        tu = {}

        def compare():
            c = 0
            f = 0
            for model in models:
                cv_score = cross_val_score(model, f,g, cv=5)
                mean = sum(cv_score) / len(cv_score)
                mean = mean * 100
                mean = round(mean, 2)
                if c < mean:
                    c = mean
                    f = models.index(model)
                tu[model] = mean
                # print('cross validation accuracies for this ', model, "=", cv_score)
                # print("accuracy of the ", model, "=", mean, "%")
                # print("<---------------------------------------------------------->")
            p.append(models[f])
            # print("the best model for the Given Dataset is ",models[f],"with the accuracy of ",c)

        compare()
        mm= p[0]
        if mm.fit(qq,ee):
            # print("model is ready")
            pass
        m = mm.predict(ww)
        n = accuracy_score(rr, m)
        return tu,mm,n
    else:
        tt=vvv[t[-1]]
        models = [LogisticRegression(max_iter=10000), SVC(), RandomForestClassifier(random_state=0), XGBClassifier()]
        
        mm=models[tt]
        mm.fit(qq,ee)
        m = mm.predict(ww)
        n=accuracy_score(rr,m)
        return mm,n












