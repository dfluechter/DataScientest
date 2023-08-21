import streamlit as st

st.title("Modelisation")

st.write("""
# Explore different classifier
Which one is the best?
""")

dataset_name = st.sidebar.selectbox("Select Dataset", ("Iris", "Breast Cancer", "Wine"))

st.write(dataset_name) 