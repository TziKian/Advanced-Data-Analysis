import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt


def run_streamlit_app():
    st.title("Cleaned Customer Dashboard")
    
    #Create sample cleaned data
    data = pd.DataFrame({
        "Gender": ["Female", "Male", "Female", "Male", "Female"],
        "City": ["Kuala Lumpur", "Kuala Lumpur", "Penang", "Johor Bahru", "Johor Bahru"],
        "Spend": [120.5, 98.0, 110.0, 105.0, 850.0]
    })
    
    #Cleaned Record Count
    st.metric(label="Total Cleaned Records", value=len(data))
    
    #Categorical Filter
    city_filter = st.selectbox("Filter by City", options=["All"] + list(data['City'].unique()))
    
    if city_filter != "All":
        filtered_data = data[data['City'] == city_filter]
    else:
        filtered_data = data
        
    st.write(filtered_data)
    
    #Chart
    st.subheader("Spend Distribution by Gender")
    fig, ax = plt.subplots()
    filtered_data.groupby("Gender")["Spend"].sum().plot(kind="bar", ax=ax, color=['#1f77b4', '#ff7f0e'])
    ax.set_ylabel("Total Spend")
    st.pyplot(fig)

run_streamlit_app() 