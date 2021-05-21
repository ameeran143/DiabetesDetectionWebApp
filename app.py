# This program detects if someone has diagbeted using machine learning and python!

import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from PIL import Image
import streamlit as st

st. set_page_config(layout="wide")

# Create a title and a sub title
st.write("""
# Diabetes Detection
Detect if someone has diabetes using machine learning and python!
""")

# Open and display and image
Image = Image.open('diabetesdetection.png')
st.image(Image, caption= "ML Project picture", use_column_width= True)

# Reading in the Data
data = pd.read_csv("diabetes.csv")
data_on_chart = pd.read_csv("diabetes.csv", nrows= 10)

#Setting a subheader in the Webapp
st.subheader("Data Information:")

# Showing the Data as a Table
st.dataframe(data)

# Showing the data statistics
st.write(data.describe())

#Showing the data as a chart
chart = st.bar_chart(data_on_chart)

# Spltting the Data into x and y : means all rows
X = data.iloc[:, 0:8] # Features
Y = data.iloc[:, -1]  # Labels

# Splitting into testing and training sets
X_train, X_Test, Y_train, Y_Test = train_test_split(X,Y, test_size=0.25, random_state=0)

# Creating and Training the Model
RandomForestClassifier = RandomForestClassifier()
RandomForestClassifier.fit(X_train, Y_train)

# Getting User input - Feature input, creating sliders to store the input
def get_user_input():
    st.sidebar.write("""**Enter Your Information**""")
    pregnancies =  st.sidebar.slider("Pregnancies (Number of times pregnant)", 0, 17, 3)
    glucose = st.sidebar.slider("Glucose (Plasma glucose concentration a 2 hours in an oral glucose tolerance test)", 0,199,117)
    blood_pressure = st.sidebar.slider("Blood Pressure (Diastolic blood pressure (mm Hg))", 0, 122, 77)
    skin_thickness = st.sidebar.slider("Skin Thickness (Triceps skin fold thickness (mm))", 0,17,3)
    insulin = st.sidebar.slider('Insulin (2-Hour serum insulin (mu U/ml))', 0.0, 846.0, 30.0)
    BMI = st.sidebar.slider("BMI (weight in kg/(height in m)^2)", 0.0, 67.1, 32.0)
    DPF = st.sidebar.slider("DPF (Diabetes pedigree function)", 0.078, 2.42, 0.3725)
    Age = st.sidebar.slider("Age (years)", 21, 81, 29)

# Store dictionary into a variable
    user_data = {
        'pregnancies' : pregnancies,
        'glucose':glucose,
        'blood_pressure':blood_pressure,
        'skin_thickness':skin_thickness,
        'insulin':insulin,
        'BMI':BMI,
        'DPF':DPF,
        'age':Age
    }

    # Transform data into a dataframe called Features
    features =pd.DataFrame(user_data, index=[0])
    return features


# create a variable to store user input
user_input = get_user_input()

# Setting a Subheader and then displaying User Input
st.subheader("User Input:")
st.write(user_input)



#Show the Model Metrics
st.subheader("Model Test Accuracy Score")
st.write(str(accuracy_score(Y_Test, RandomForestClassifier.predict(X_Test)) * 100) + '%')

# Store Model Prediction in a variable
prediction = RandomForestClassifier.predict(user_input)
if prediction == 1:
    prediction = "The model has classified you as are diabetic"
else:
    prediction = "The model has classified you as not diabetic"

# Setting a Subheader to display claassifciaiton
st.write("""# Model Prediction""")
st.write(f"""
{prediction}""")
