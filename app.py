# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 20:47:59 2021

@author: Mandar Joshi
"""

import streamlit as st
import spotipy
import spotipy.oauth2 as oauth2
import pickle


model2 = pickle.load(open('spotify_model.pkl','rb'))


client_credentials_manager = oauth2.SpotifyClientCredentials(client_id='5e22c02862e0444aac0351ee2aca9exx', client_secret='dd2407b5b3984fa38bf707b883e34bxx')
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

user_input_song_id = st.text_input("Enter song ID", '5eflbiBkMjtAlgkjiPXU9A')


aud_features  = sp.audio_features('6ZKtNuLLVIOlcR9si8JFHp')

danceability =aud_features[0]['danceability']
loudness = aud_features[0]['loudness']
speechiness =aud_features[0]['speechiness']
acousticness = aud_features[0]['acousticness']
liveness =aud_features[0]['liveness']
instrumentalness =aud_features[0]['instrumentalness']

x = model2.predict([[danceability,loudness,speechiness,acousticness,liveness,instrumentalness]])
print(x)

if x == 1:
    st.text('Lofi')
elif x == 2:
    st.text('Dance')
else:
    st.text('Rock')




