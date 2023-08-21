import streamlit as st
from sklearn import datasets
st.title("Modelisation")

st.write("""
# Explore different classifier
Which one is the best?
""")

dataset_name = st.sidebar.selectbox("Select Dataset", ("Iris", "Breast Cancer", "Wine"))

classifier_name = st.sidebar.selectbox("Select Classsifier", ("KNN", "SVM", "Logistic Regression"))

def get_dataset(dataset_name):
    if dataset_name == "Iris":
        data = datasets.load_iris()
    elif dataset_name == "Wine":
        data = datasets.load_wine()
    else:
        data = datasets.load_breast_cancer()
    X = data.data
    y = data.target
    return X, y

X, y = get_dataset(dataset_name)
st.write(dataset_name) 