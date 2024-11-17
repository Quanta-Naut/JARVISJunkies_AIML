import streamlit as st
import pickle
import pandas as pd

# Load the pickle file
@st.cache_data
def load_model():
    with open('best_model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

# Load the model
model = load_model()

# Streamlit app interface
st.title("Pickle Model Deployment Example")

# Input for prediction
user_id = input("Enter User ID: ")
age = int(input("Enter Age: "))
gender = input("Enter Gender (Male/Female): ")
platform = input("Enter Platform (Whatsapp/Instagram...etc): ")
daily_usage_time = int(input("Enter Daily Usage Time (in minutes): "))
posts_per_day = int(input("Enter Posts Per Day: "))
likes_received_per_day = int(input("Enter Likes Received Per Day: "))
comments_received_per_day = int(input("Enter Comments Received Per Day: "))
messages_sent_per_day = int(input("Enter Messages Sent Per Day: "))

user_input = pd.DataFrame({
    'User_ID': [user_id],
    'Age': [age],
    'Gender': [gender],
    'Platform': [platform],
    'Daily_Usage_Time (minutes)': [daily_usage_time],
    'Posts_Per_Day': [posts_per_day],
    'Likes_Received_Per_Day': [likes_received_per_day],
    'Comments_Received_Per_Day': [comments_received_per_day],
    'Messages_Sent_Per_Day': [messages_sent_per_day]
})

# Predict using the model
if st.button("Predict"):
    result = model.predict([[user_input]])
    st.write(f"Prediction: {result[0]}")
