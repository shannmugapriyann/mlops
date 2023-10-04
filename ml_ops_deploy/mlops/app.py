import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import pickle
st.set_option('deprecation.showPyplotGlobalUse', False)
st.title("Web Deployment of Medical Diagnostic App ")
st.subheader("Is the person diabetic ?")
df=pd.read_csv("diabetes.csv")
if st.sidebar.checkbox("View Data",False):
    st.write(df)
if st.sidebar.checkbox("View Distributions",False):
    df.hist()
    plt.tight_layout()
    st.pyplot()

# Step 1: Load the pickled model
model=open("rfc.pickle","rb")
clf=pickle.load(model)
model.close()

# Step 2: Get the front end user input
pregs=st.number_input('Pregnancies',0,17,0)
glucose=st.number_input('Glucose',44,199,44)
BP=st.number_input( 'BloodPressure',24,122,24)
Skin=st.slider('SkinThickness',7,99,7)
Insulin=st.slider('Insulin',14,846,14)
BMI=st.slider('BMI',18,67,18)
dpf=st.slider('DiabetesPedigreeFunction',0.05,2.50,0.05) 
Age=st.slider('Age',21,85,21)

# Step 3: Convert user input to model input
data={'Pregnancies':pregs, 'Glucose':glucose, 'BloodPressure':BP, 'SkinThickness':Skin, 'Insulin':Insulin,
       'BMI':BMI, 'DiabetesPedigreeFunction':dpf, 'Age':Age}
input_data=pd.DataFrame([data])

# Step 4: Get the predictions and print the result
prediction=clf.predict(input_data)[0]
if st.button("Predict"):
    if prediction==1:
        st.subheader('Diabetic')
    else:
        st.subheader('Healthy')
