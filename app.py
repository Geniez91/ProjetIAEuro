import streamlit as st
import requests
import pandas as pd
from datetime import datetime

st.set_page_config(
    page_title="Streamlit App",
    page_icon="",
    layout="wide",
)

st.title("Goals prediction")

st.image('https://www.francebleu.fr/s3/cruiser-production/2019/03/a98102ae-5e60-4558-b404-f43cfab46b86/1200x680_coupe-du-monde.jpg')

st.header("Train Model")
# Form for training data
with st.form(key='train_form'):
    train_button = st.form_submit_button(label='Train')

# If train button is clicked, send request to API to train the model
if train_button:
    response = requests.post('http://localhost:8000/train/')
    if response.status_code == 200:
        st.write(response.json())
    else:
        st.write("Error: API request failed.")


st.header("Predict Model")
# Form for input data
with st.form(key='input_form'):
    game_id = st.number_input('game_id')
    player_id = st.number_input('player_id')
    player_club_id = st.number_input('player_club_id')
    player_current_club_id = st.number_input('player_current_club_id')
    yellow_cards = st.number_input('yellow_cards')
    red_cards = st.number_input('red_cards')
    assists = st.number_input('assists')
    minutes_played = st.number_input('minutes_played')
   
    submit_button = st.form_submit_button(label='Submit')

# If form is submitted, send data to API for prediction
if submit_button:
    data = [
        game_id,
        player_id,
        player_club_id,
        player_current_club_id,
        yellow_cards,
        red_cards,
        assists,
        minutes_played
    ]
    response = requests.post('http://localhost:8000/predict/', json=[data])
    if response.status_code == 200:
        st.write(response.json())
    else:
        st.write("Error: API request failed.")

with open('models/model.pkl', 'rb') as f:
   st.download_button('Download CSV', f, file_name='model.pkl') 