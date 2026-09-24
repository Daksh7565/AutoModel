# -*- coding: utf-8 -*-
"""
Created on Sat Feb 15 23:32:05 2025

@author: Lenovo
"""
import pickle as pk
import  MachineTraining as ML
import streamlit as st 
import time
import pandas as pd

st.header("Predictive system")
st.sidebar.header("Dropdown Menu")
options = ["Home","Text", "NLP", "Image"]
selected_option = st.sidebar.selectbox("Choose an option", options)
if selected_option==options[0]:
    st.write("the predictive system is used to run file ")
    
elif selected_option==options[1]:
    uploaded_file = st.file_uploader("Upload your dataset (CSV file)", type=["csv"])

    if uploaded_file is None:
        st.info("Upload a CSV file to begin.")
        st.stop()

    # Save the upload into the Dataset folder so MachineTraining can read it
    a="/"+uploaded_file.name
    with open(ML.DATASET_DIR + a, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.write("File selected:", a)
    if st.button("Information of dataset"):
        c=ML.info(a)
        st.write("the number of rows:",c[0])
        st.write("the number of columns:",c[1])
        st.write("the name of each attributes are ",c[2])
        st.balloons()
    b=st.text_input("enter the output attribute for the dataset")
    if st.button("Analysis of dataset"):
        with st.spinner("Analyzing dataset... ⏳"):
            progress_bar = st.progress(0)

            # Simulating training steps
            for percent_complete in range(0, 101, 20):  # Increments of 20%
                time.sleep(1)  # Simulate training time
                progress_bar.progress(percent_complete)
            k=ML.check(a,b)
        if k[0]==1:
            st.write("the Dataset is Regression Output Values")
        else:
            st.write("the Dataset is Classification Output Values")
        if k[1]==0:
            st.write("The Dataset have no Null values")
        else:
            st.write("The Dataset have Null values")
        if k[2]==0:
            st.write("The Dataset don,t required LabelEncoder as all the Data is in integer and float format ")
        else:
            st.write("The Dataset have Object as datatype requierd LabelEncoder")
        if k[3]==0:
            st.write("The Dataset is balanced and evenly distrubted. ")
        else:
            st.write("The Dataset is not  balanced and not evenly distrubted. ")
            st.write("The Dataset has  classification class are ")
            u=k[4].keys()
            i=k[4].values()
            ii=pd.DataFrame({"Class" : u , "Value" : i} )
            st.dataframe(ii)
    opt1=["Select the option","Manual","Recommended"]
    choice=st.selectbox("Select the type of training u want to acquire",options=opt1)
    if choice=="Manual":
        p = st.radio(
        "How do you want to remove null values?",
        ("Dropping the row", "Applying imputation on the data"))
        o = st.radio(
        "Do you want to perform Data Imbalancing?",
        ("Yes", "No"))
        u=st.slider("Select the Size of Test Data",0.0,1.0,0.2,0.05)
        y=st.radio("How do u want to select Model for training? ",("Recommend the Best model","select the Specific model"))
        if y=="select the Specific model":
            t=st.radio("Select the Model u want to use ",("Logistic Regression","RandomForest","SVC","XGBoost"))
            qqqq=st.radio("Do u want perform hyper parameter tuning before training model:",("Yes","No"))
            model_hper=[]
            if qqqq=="Yes":
                model_hper=[t]
                if t == "Logistic Regression":
                    st.markdown("### 🔹 Logistic Regression Hyperparameters")
                    st.write("""
                    - **C**: Inverse of regularization strength. Smaller values → stronger regularization.  
                    - **penalty**: Type of regularization (`l1`, `l2`, `elasticnet`).  
                    - **solver**: Optimization algorithm used (liblinear, saga, lbfgs, newton-cg).  
                    - **max_iter**: Maximum number of iterations before stopping.
                    """)
        
                    C = st.slider("The value of C", 0.001, 10.0, 1.0, 0.1)
                    penalty = st.radio("The value of penalty is", ("l1", "l2", "elasticnet"))
                    solver = st.radio("Solver", ("liblinear", "saga", "lbfgs", "newton-cg"))
                    max_iter = st.slider("Max iterations", 50, 1000, 200, 50)
                    model_hper.append(C)
                    model_hper.append(penalty)
                    model_hper.append(solver)
                    model_hper.append(max_iter)
                    
        
                # ---------------- Random Forest ----------------
                elif t == "RandomForest":
                    st.markdown("### 🌲 Random Forest Hyperparameters")
                    st.write("""
                    - **n_estimators**: Number of trees in the forest.  
                    - **max_depth**: Maximum depth of each tree.  
                    - **min_samples_split**: Minimum samples required to split an internal node.  
                    - **min_samples_leaf**: Minimum samples required at a leaf node.  
                    - **criterion**: Function used to measure split quality (`gini`, `entropy`, `log_loss`).
                    """)
        
                    n_estimators = st.slider("Number of Trees (n_estimators)", 10, 500, 100, 10)
                    max_depth = st.slider("Max Depth", 1, 50, 10, 1)
                    min_samples_split = st.slider("Min Samples Split", 2, 10, 2, 1)
                    min_samples_leaf = st.slider("Min Samples Leaf", 1, 10, 1, 1)
                    criterion = st.radio("Criterion", ("gini", "entropy", "log_loss"))
                    model_hper.append(n_estimators)
                    model_hper.append(max_depth)
                    model_hper.append(min_samples_split)
                    model_hper.append(min_samples_leaf)
                    model_hper.append(criterion)
        
                # ---------------- Support Vector Classifier ----------------
                elif t == "SVC":
                    st.markdown("### ⚡ Support Vector Classifier (SVC) Hyperparameters")
                    st.write("""
                    - **C**: Regularization strength. Higher → less regularization.  
                    - **kernel**: Type of kernel function (`linear`, `poly`, `rbf`, `sigmoid`).  
                    - **degree**: Degree of polynomial kernel (if `poly`).  
                    - **gamma**: Kernel coefficient (`scale` or `auto`).
                    """)
        
                    C = st.slider("The value of C", 0.01, 10.0, 1.0, 0.1)
                    kernel = st.radio("Kernel", ("linear", "poly", "rbf", "sigmoid"))
                    degree = st.slider("Degree (for poly kernel)", 2, 10, 3, 1)
                    gamma = st.radio("Gamma", ("scale", "auto"))
                    model_hper.append(C)
                    model_hper.append(kernel)
                    model_hper.append(degree)
                    model_hper.append(gamma)
        
                # ---------------- XGBoost ----------------
                elif t == "XGBoost":
                    st.markdown("### 🚀 XGBoost Hyperparameters")
                    st.write("""
                    - **n_estimators**: Number of boosting rounds (trees).  
                    - **learning_rate**: Step size shrinkage to prevent overfitting.  
                    - **max_depth**: Maximum depth of a tree.  
                    - **subsample**: Fraction of samples used per tree.  
                    - **colsample_bytree**: Fraction of features used per tree.
                    """)
        
                    n_estimators = st.slider("Number of Trees (n_estimators)", 50, 500, 100, 50)
                    learning_rate = st.slider("Learning Rate", 0.01, 0.5, 0.1, 0.01)
                    max_depth = st.slider("Max Depth", 1, 15, 6, 1)
                    subsample = st.slider("Subsample", 0.1, 1.0, 0.8, 0.1)
                    colsample_bytree = st.slider("Colsample By Tree", 0.1, 1.0, 0.8, 0.1)
                    model_hper.append(n_estimators)
                    model_hper.append(learning_rate)
                    model_hper.append(max_depth)
                    model_hper.append(subsample)
                    model_hper.append(colsample_bytree)
            else:
                model_hper=[t]
                        
        else:
            t=0
            model_hper=[]
        if st.button("Predictive System"):
            with st.spinner("Training the model... ⏳"):
                progress_bar = st.progress(0)

                # Simulating training steps
                for percent_complete in range(0, 101, 20):  # Increments of 20%
                    time.sleep(1)  # Simulate training time
                    progress_bar.progress(percent_complete)

                c = ML.ttrain(a, b,p,o,u,y,model_hper)
                st.write("The model is ",c[0])
                st.write("The accuracy of this model is ",c[1])
    elif choice=="Recommended":
        if st.button("Predictive system:"):
            # k = ML.check(a, b)
            # if k[1]==1:
            #
            #     st.write("The Dataset has Null Values")
            #     st.write("The total null value is ",sum(k[5].values()))
            #     st.dataframe({"columns":k[5].keys(),"null value ": k[5].values()})
            #
            #     option=["Remove Null Value","Imputation"]
            #     opt=st.selectbox("Select the method to remove null value ", option)
            #     if opt=="Remove Null Value":
            #         ss=ML.null(a,0)
            #     else:
            #         ss=ML.null(a,1)
            #

            with st.spinner("Training the model... ⏳"):
                progress_bar = st.progress(0)

                # Simulating training steps
                for percent_complete in range(0, 101, 20):  # Increments of 20%
                    time.sleep(1)  # Simulate training time
                    progress_bar.progress(percent_complete)

                c = ML.op(a, b)

            progress_bar.empty()  # Remove progress bar once training is done
            st.success("Model training completed! 🎉")
            st.write("the atrribute of the dataset are:",c[0])
            st.write("the number of row and columns are:",c[1],c[2])
            st.write("the model with highest accuracy is ",c[3])
            st.write("final accuracy of model after training is:",c[4])
            st.write("the accuracy achive by each model during  cross val score are ")
            m=[]
            n=[]
            for i in c[5]:
                m.append(i)
                n.append(c[5][i])
            k={"Models":m,"Accuracy":n}
            results_df=pd.DataFrame(k)
            st.dataframe(results_df)
            st.balloons()
            with open(c[6], "rb") as file:
                st.download_button("Download Trained Model", file, file_name="best_model.sav")
            with open(c[7], "rb") as file:
                st.download_button("Download Modified Dataset", file, file_name="modified_dataset.csv")
elif selected_option==options[2]:
    a=st.text_input("Enter the name of file: ")
    option2=[]
    k=0
    option=["TFID","BOW"]
    if st.button("Information of dataset"):
        c=ML.info(a)
        k+=1
        options=c[2]
        st.write("the number of rows:",c[0])
        st.write("the number of columns:",c[1])
        st.write("the name of each attributes are ",c[2])
        st.balloons()
    b = st.text_input("Select the input attribute")
    e = st.selectbox("Select the tokenizer", options=option)
    d = st.text_input("select the output attribute")
    #b = st.selectbox("Select the input attribute", options=option2)
    #d = st.selectbox("Select the output attribute", options=option2)
    if st.button("train model"):
        c=ML.NLP(a,b,d,e)
        st.write("the name of each model used are  ",c[0])
        st.write("the model with highest accuracy is ",c[1])
        st.write("final accuracy of model after training is:",c[2])
        for i in c[3]:
            st.write(i,c[3][i])
        st.balloons()
        with open(c[4], "rb") as file:
            st.download_button("Download Trained Model", file, file_name="best_model.sav")
elif selected_option==options[3]:
    a=st.text_input("Enter the name of file: ")
    b=st.text_input("enter the name of project")
    option3=["CNN","Resnet50","Mobilenet","Hybrid"]
    d=st.selectbox("select the model",options=option3)
    if d:
        st.write("model selected is ",d)
    if "image_data" not in st.session_state:
        st.session_state.image_data = None
    if "model_data" not in st.session_state:
        st.session_state.model_data = None

    if st.button("Basic Information"):
        c = ML.image(a, b,d)
        st.session_state.image_data = c[0]  # Store folder details
        st.session_state.model_data = c[1]  # Store model details

    # Display stored folder details (if available)
    if st.session_state.image_data:
        q = st.session_state.image_data
        st.write("### Folder/Dataset Details:")
        st.write(f"🔹 **Data Split Ratio:** {q[0]} (Train, Test, Validation)")
        st.write(f"📂 **Dataset Path:** {q[1]}")
        st.write(f"🖼️ **static Resize Shape:** {q[2]}")
        st.write(f"📦 **Batch Size:** {q[3]}")

    # Display stored model details (if available)
    if st.session_state.model_data:
        w = st.session_state.model_data
        st.write("### Model Details:")
        st.write(f"📌 **Loss Function:** {w[1][0]}")
        st.write(f"⚡ **Optimizer:** {w[1][1]}")
        st.write(f"📊 **Metrics:** {w[1][2]}")
        st.write(f"⏳ **Epochs:** {w[2]}")
        st.write(f"💾 **Model File:** {w[0]}")

        # Load model for accuracy and structure display
        model = pk.load(open(w[0], 'rb'))

        if st.button("Model Accuracy & Structure"):
            st.write("### Model Summary:")
            import io
            import sys
            
            # Capture model summary as a string
            summary_buffer = io.StringIO()
            sys.stdout = summary_buffer
            model.summary()
            sys.stdout = sys.__stdout__  # Reset stdout
            
            st.text("### Model Summary:")
            st.text(summary_buffer.getvalue())  # Display captured model summary
            st.write("📈 **Model Accuracy:**", w[3].history['accuracy'][-1])
            st.write("📈 **Model Loss:**", w[3].history['loss'][-1])
    #q=[]
    #w=[]

    #if st.button("basic information"):
    #    c=ML.image(a,b)
    #   x=len(c[0])
    #    for i in c[0]:
    #        q.append(i)
    #    y=len(c[1])
     #   for i in c[1]:
      #      w.append(i)
       # model=pk.load(open(w[0],'rb'))
        #st.balloons()
        #if st.button("model accuracy and structure"):
         #   st.write("the structure of model is:",model.summary())
          #  st.write("the accuracy of model is ",w[3]['accuracy'])
    #if len(q)!=0:
     #   st.write("the detail regarding the folder/Dataset are:")
      #  st.write("the ratio of spliting data in ",q[0],"train,test,validation respectively")
       # st.write("the path of the file:",q[1])
        #st.write("the image are reshape in ",q[2])
        #st.write("the size of each bztch is:",q[3])
    #if len(w)!=0:
     #   st.write("the model loss function used in model is :",w[1][0])
      #  st.write("the optimizer used in model is :",w[1][1])
       # st.write("the metrics used in model is :",w[1][2])
        #st.write("the number of epochs are",w[2])
        #st.write("the name of model file is ",w[0])
    #st.write(w)
    
    
    
    
    
    
        