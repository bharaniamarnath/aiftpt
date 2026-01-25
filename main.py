import streamlit as st
import pandas as pd
from process import load_data
from analyze import get_basic_info, top_nationalities
from visualize import plot_rating_distribution, plot_top_players
from cluster import perform_clustering

# Load data
data_url = 'datasets/players_20.csv'
data = load_data(data_url)

st.title('FIFA 20 Players Analysis')

# Create tabs for different functionalities
tab1, tab2, tab3, tab4 = st.tabs(["Basic Info", "Top Nationalities", "Visualizations", "Perform Clustering"])

# Basic Info Tab
with tab1:
    if st.button("Show Basic Info"):
        head, info, description = get_basic_info(data)
        st.write(head)
        st.write(description)

# Top Nationalities Tab
with tab2:
    if st.button("Top Nationalities"):
        top_natio = top_nationalities(data)
        st.write(top_natio)

# Visualizations Tab
with tab3:
    if st.button("Show Rating Distribution"):
        plot_rating_distribution(data)
    if st.button("Show Top Players"):
        plot_top_players(data)

# Clustering Tab
with tab4:
    if st.button("Perform Clustering"):
        clusters, centers = perform_clustering(data)
        st.write(f'Clusters: {clusters}')
        st.write(f'Cluster Centers: {centers}')