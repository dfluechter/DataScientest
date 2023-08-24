import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import randint as sp_randint, uniform as sp_randFloat
#from st_aggrid import AgGrid

# Function to load data and cache it
@st.cache_data
def load_data(data_url):
    data = pd.read_csv(data_url)
    data.drop(columns=['Unnamed: 0'], inplace=True)
    return data

# Function to create and display correlation heatmap
def create_correlation_heatmap(data):
    fig, ax = plt.subplots(figsize=(5, 2.5))
    sns.heatmap(data.corr(), linewidth=.5, annot=True, cmap="coolwarm", fmt=".1f")
    st.pyplot(fig)

# Function to create and display scatter plot
def create_scatter_plot(data):
    X = data.drop(columns=["sta"])
    y = data["sta"]
    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.25, random_state=23)
    gb_model = GradientBoostingRegressor()
    gb_model.fit(train_X, train_y)
    gb_predictions = gb_model.predict(test_X)
    fig, ax = plt.subplots(figsize=(10, 5),)
    sns.scatterplot(x=test_y, y=gb_predictions)
    st.pyplot(fig)

# Function to create and display bar plot
def create_bar_plot(data):
    fig, ax = plt.subplots(figsize=(3.5, 2))
    sns.barplot(x="year",
                y="sta",
                data=data)
    st.pyplot(fig)

# Function to create and display histogram
def create_histogram(data):
    fig, ax = plt.subplots(figsize=(5, 3))
    sns.distplot(data.sta, kde=True)
    st.pyplot(fig)

# Main function
def main():
    st.title("Web App using Streamlit")
    st.image("streamlit.png", width=500)
    st.title("World Temperature")
    
    data_url = "datasets/datas_pre_processed.csv"
    data = load_data(data_url)
    st.write("Shape of dataset:", data.shape)
    
    menu = st.sidebar.radio("Menu", ["Home", "Find Optimal Parameters", "Gradient Boosting Regressor"])
    
    if menu == "Home":
        st.header("STA")
        
        if st.checkbox("Tabular Data"):
            #st.table(data.head(150))
            AgGrid(data.head(150))
        st.header("Statistical Summary of the Dataframe")
        if st.checkbox("Statistics"):
            st.table(data.describe())
        
        st.header("Correlation Graph")
        if st.checkbox("Correlation Heatmap"):
            create_correlation_heatmap(data)
        
        st.title("Graphs")
        graph = st.selectbox("Different types of graphs", ["Scatter Plot", "Bar Graph", "Histogram"])
        
        if graph == "Scatter Plot":
            create_scatter_plot(data)
        elif graph == "Bar Graph":
            create_bar_plot(data)
        elif graph == "Histogram":
            create_histogram(data)
    
    if menu == "Find Optimal Parameters":
        st.title("Find Optimal Parameters")
        X = data.drop(columns=["sta"])
        y = data["sta"]
        train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.25, random_state=23)
        model = GradientBoostingRegressor()
        
        tab1, tab2 = st.tabs(["RandomizedSearch", "GridSearch"])
        
        with tab1:
            st.title("RandomizedSearch")
            parameters = {
                'learning_rate': sp_randFloat(),
                'subsample': sp_randFloat(),
                'n_estimators': sp_randint(100, 1000),
                'max_depth': sp_randint(4, 10),
            }
            with st.spinner("Running RandomizedSearchCV..."):
                randm = RandomizedSearchCV(estimator=model, param_distributions=parameters, cv=2, n_iter=10, n_jobs=-1)
                randm.fit(train_X, train_y)
            st.write("Results of RandomizedSearchCV")
            st.write("The best Estimator is", randm.best_estimator_)
            st.write("The best score is", randm.best_score_)
            st.write("The best parameters are", randm.best_params_)
        
        with tab2:
            st.title("GridSearch")
            parameters = {
                'learning_rate': [0.01, 0.02, 0.03],
                'subsample': [0.9, 0.5, 0.2],
                'n_estimators': [100, 500, 1000],
                'max_depth': [4, 6, 8]
            }
            with st.spinner("Running GridSearchCV..."):
                grid = GridSearchCV(estimator=model, param_grid=parameters, cv=2, n_jobs=-1)
                grid.fit(train_X, train_y)
            st.write("Results from Grid Search")
            st.write("The best Estimator is", grid.best_estimator_)
            st.write("The best score is", grid.best_score_)
            st.write("The best parameters are", grid.best_params_)
    
    if menu == "Gradient Boosting Regressor":
        st.title("Gradient Boosting Regressor")
        SEED = 23
        X = data.drop(columns=["sta"])
        y = data["sta"]
        train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.25, random_state=SEED)
        gbr = GradientBoostingRegressor(
            loss='squared_error',
            learning_rate=0.1,
            n_estimators=300,
            max_depth=5,
            random_state=SEED,
            max_features=11
        )
        gbr.fit(train_X, train_y)
        
        pred_y = gbr.predict(test_X)
        st.write(f"The score for the train set in Gradient Boosting Regressor model is", gbr.score(train_X, train_y))
        st.write(f"The score for the test set in Gradient Boosting Regressor model is", gbr.score(test_X, test_y))
        gb_mae = mean_absolute_error(test_y, pred_y)
        gb_mse = mean_squared_error(test_y, pred_y)
        gb_r2 = r2_score(test_y, pred_y)
        st.write(f"Mean Absolute Error: {gb_mae:.2f}")
        st.write(f"Mean Square Error: {gb_mse:.2f}")
        st.write(f"R2: {gb_r2:.2f}")
        
        st.header("Predict STA")
        #year_input = st.number_input("Enter a year:", min_value=int(data["year"].min()), max_value=int(data["year"].max()), step=1)
        year_input = st.number_input("Enter a year:", min_value="2020", max_value="2030", step=1)
        
        avg_values = data.drop(columns=["sta", "year"]).mean()
        prediction_input = pd.DataFrame({
            "country_id": [0],
            "year": [year_input]
        })
        
        for column, value in avg_values.items():
            prediction_input[column] = value
        
        prediction = gbr.predict(prediction_input)[0]
        st.write(f"Predicted STA for the year {year_input}: {prediction:.2f}")

if __name__ == "__main__":
    main()
