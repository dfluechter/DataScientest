import numpy as np
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
st.write("shape of dataset", X.shape)
st.write("number of classes", len(np.unique(y)))
 
def add_parameter_ui(clf_name):
    params = dict()
    if clf_name == "KNN":
        K = st.sidebar.slider("K", 1, 15)
        params["K"] = K
    elif clf_name == "SVM":
        C = st.sidebar.slider("C", 0.01, 10.0)
        params["C"] = C
    else:
        solver = st.sidebar.selectbox("Solver", ("newton-cg", "lbfgs", "liblinear"))
        C = st.sidebar.slider("C", 0.01, 10.0)
        params["solver"] = solver
        params["C"] = C
    return params
add_parameter_ui(classifier_name)