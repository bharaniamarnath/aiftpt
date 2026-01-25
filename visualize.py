import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

def plot_rating_distribution(data):
    plt.figure(figsize=(10, 6))
    sns.histplot(data['overall'], bins=30, color='blue', kde=True)
    plt.title('Distribution of Overall Ratings')
    plt.xlabel('Overall Rating')
    plt.ylabel('Frequency')
    st.pyplot(plt)  # Render the plot in Streamlit
    plt.clf()  # Clear the figure after rendering

def plot_top_players(data):
    top_players = data.sort_values(by='overall', ascending=False).head(10)
    plt.figure(figsize=(12, 8))
    sns.barplot(x='overall', y='short_name', data=top_players, palette='viridis')
    plt.title('Top 10 Players by Overall Rating')
    plt.xlabel('Overall Rating')
    plt.ylabel('Player')
    st.pyplot(plt)  # Render the plot in Streamlit
    plt.clf()  # Clear the figure after rendering

# Add similar adjustments to other plotting functions...
