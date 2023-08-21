#import pygwalker as pyg
import pandas as pd
#import streamlit.components.v1 as components
import streamlit as st


header = st.container()
dataset = st.container()
features = st.container()
model_training = st.container() 

with header:
    st.text('In this projekt')
    st.title('Welcome to Streamlit!')

with dataset:
    st.text('NYC dataset')
    st.title('In this projekt....')

    dataset = pd.read_csv('./data/.csv')
    st.write(dataset.head())

    pulocation_dist = pd.DataFrame(dataset['PULocationID'].value_counts())
    st.bar_chart(pulocation_dist)

with features:
    st.header('Features')

with model_training:
    st.header('Time to train the model')
    st.text('Here you get to choose the hyperparameters of the model')