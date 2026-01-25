#!/usr/bin/env python
# coding: utf-8

# # PTID-CDS-DEC-25-3624_PRCP-1004-Fifa20

# In[1]:


# import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


# configurations
import warnings
warnings.filterwarnings("ignore")

# seaborn plot style
sns.set(style="whitegrid")


# ## Data analysis

# In[3]:


# load dataset
url = 'datasets/players_20.csv'  # Update with the correct path
data = pd.read_csv(url)


# In[4]:


# show first 5 rows
print("First five rows of the dataset:")
print(data.head())


# In[5]:


# dataset basic info
print("\nDataset information:")
print(data.info())


# In[6]:


# show summary stats
print("\nSummary statistics:")
print(data.describe())


# In[7]:


# overall ratings distribution
plt.figure(figsize=(10, 6))
sns.histplot(data['overall'], bins=30, color='blue', kde=True)
plt.title('Distribution of Overall Ratings')
plt.xlabel('Overall Rating')
plt.ylabel('Frequency')
plt.show()


# In[8]:


# top 10 players by overall ratings
top_players = data.sort_values(by='overall', ascending=False).head(10)
plt.figure(figsize=(12, 8))
sns.barplot(x='overall', y='short_name', data=top_players, palette='viridis', hue=None, legend=False)
plt.title('Top 10 Players by Overall Rating')
plt.xlabel('Overall Rating')
plt.ylabel('Player')
plt.show()


# In[9]:


# top 10 players by nationality
top_nationalities = data['nationality'].value_counts().head(10)
plt.figure(figsize=(10, 6))
top_nationalities.plot(kind='bar', color='orange')
plt.title('Top 10 Nationalities of Players')
plt.xlabel('Nationality')
plt.ylabel('Number of Players')
plt.xticks(rotation=45)
plt.show()


# In[10]:


# heatmap - correlation of numerical attr
numeric_data = data.select_dtypes(include=[np.number])
plt.figure(figsize=(12, 10))
correlation_matrix = numeric_data.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Correlation Heatmap of Attributes')
plt.show()


# In[11]:


# market value vs overall rating
plt.figure(figsize=(10, 6))
sns.scatterplot(x='overall', y='value_eur', data=data, alpha=0.5)
plt.title('Market Value vs Overall Rating')
plt.xlabel('Overall Rating')
plt.ylabel('Market Value (in Euros)')
plt.xlim(40, 100)
plt.ylim(0, 200000000)
plt.show()


# ## Clustering

# In[12]:


from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


# In[13]:


# EDA on key football skills
attributes = ['pace', 'shooting', 'passing', 'dribbling', 'defending', 'overall']
cdata = data[attributes].copy()


# In[14]:


# check missing values
print(cdata.isnull().sum())


# In[15]:


# fill missing values with mean val
cdata.fillna(cdata.mean(), inplace=True)


# In[16]:


# skill distributions plot
plt.figure(figsize=(12, 8))
for i, column in enumerate(cdata.columns, 1):
    plt.subplot(2, 3, i)
    sns.histplot(cdata[column], bins=30, kde=True)
    plt.title(f'Distribution of {column}')
    plt.xlabel(column)
    plt.ylabel('Frequency')

plt.tight_layout()
plt.show()


# In[17]:


# normalize data
scaler = StandardScaler()
cdata_scaled = scaler.fit_transform(cdata)


# In[18]:


# k means clustering
inertia = []
ck_values = range(1, 11)
for ck in ck_values:
    kmeans = KMeans(n_clusters=ck, random_state=42)
    kmeans.fit(cdata_scaled)
    inertia.append(kmeans.inertia_)

# elbow method plot
plt.figure(figsize=(10, 6))
plt.plot(ck_values, inertia, marker='o')
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.xticks(ck_values)
plt.grid()
plt.show()


# In[19]:


# Assume optimal k to be 5
optimal_ck = 5
kmeans = KMeans(n_clusters=optimal_ck, random_state=42)
clusters = kmeans.fit_predict(cdata_scaled)

# add cluster info to the original data
cdata['Cluster'] = clusters

# plot clusters using PCA
pca = PCA(n_components=2)
pca_result = pca.fit_transform(cdata_scaled)
cdata['PCA1'] = pca_result[:, 0]
cdata['PCA2'] = pca_result[:, 1]

plt.figure(figsize=(10, 6))
sns.scatterplot(x='PCA1', y='PCA2', hue='Cluster', data=cdata, palette='viridis', alpha=0.7)
plt.title('PCA of Player Skills Clusters')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.legend(title='Cluster')
plt.show()


# In[20]:


# show cluster centers in original space
cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)
cluster_centers_data = pd.DataFrame(data=cluster_centers, columns=attributes)

# show cluster centers with player attributes
print(cluster_centers_data)


# ## top 10 countries with most players

# In[21]:


# count players by country
country_counts = data['nationality'].value_counts()

# get top 10 countries
top_countries = country_counts.head(10)

# print top 10 countries
print(top_countries)


# ## overall rating vs. age of players

# In[22]:


# sselecting req columns
age = data['age']
overall_rating = data['overall']

# create scatter plot
plt.figure(figsize=(12, 6))
plt.scatter(age, overall_rating, alpha=0.5)
plt.title('Overall Rating vs. Age of Players')
plt.xlabel('Age')
plt.ylabel('Overall Rating')
plt.grid()
plt.show()


# ## age after which a player stops improving

# In[23]:


# select req columns
# drop na values
nidata = data[['age', 'overall', 'potential']]
nidata = nidata.dropna()

# group by age
# calculate the average overall and potential
age_grouped = nidata.groupby('age').mean().reset_index()

# add new column
# check if a player is improving
age_grouped['improvement'] = age_grouped['potential'] - age_grouped['overall']


# In[24]:


# plot average overall rating and potential by age
plt.figure(figsize=(15, 6))

# plot overall vs age
plt.subplot(1, 2, 1)
sns.lineplot(x='age', y='overall', data=age_grouped, marker='o', label='Average Overall')
sns.lineplot(x='age', y='potential', data=age_grouped, marker='o', label='Average Potential')
plt.title('Average Overall and Potential Rating by Age')
plt.xlabel('Age')
plt.ylabel('Rating')
plt.legend()

# plot improvement vs age
plt.subplot(1, 2, 2)
sns.lineplot(x='age', y='improvement', data=age_grouped, marker='o')
plt.axhline(0, color='red', linestyle='--')
plt.title('Improvement (Potential - Overall) by Age')
plt.xlabel('Age')
plt.ylabel('Improvement')
plt.ylim(-10, 10)

plt.tight_layout()
plt.show()


# In[25]:


# check improvement values
print(age_grouped[['age', 'overall', 'potential', 'improvement']])

# check where players stop showing significant improvement
# set threshold for min improvement
threshold = 1

# find first occurence in age where improvement is no longer above threshold val
if not age_grouped[age_grouped['improvement'] < threshold].empty:
    age_stopping_improvement = age_grouped[age_grouped['improvement'] < threshold].age.min()
    print(f"Players typically stop showing significant improvement (less than set threshold {threshold}) at age {age_stopping_improvement}.")
else:
    print("Players continue to improve or unable to determine where improvement stops due to insufficient data points.")


# ## type of offensive players tends to get paid the most

# In[26]:


opdata = data.copy()

# split player_positions into a list
opdata['player_positions'] = opdata['player_positions'].str.split(',')


# In[27]:


# define function to categorize players into offensive types
def categorize_offensive_player(positions):
    if any(pos in positions for pos in ['ST', 'CF', 'LS', 'RS']):
        return 'Striker'
    elif any(pos in positions for pos in ['RW']):
        return 'Right Winger'
    elif any(pos in positions for pos in ['LW']):
        return 'Left Winger'
    else:
        return None


# In[28]:


# apply function to create a new column for player type
opdata['offensive_type'] = opdata['player_positions'].apply(categorize_offensive_player)

# filter out rows where offensive_type is none
offensive_players = opdata[opdata['offensive_type'].notnull()]

# group by offensive type and calculate average wage
avg_wage_by_type = offensive_players.groupby('offensive_type')['wage_eur'].mean().reset_index()


# In[29]:


# plotting the average wages
plt.figure(figsize=(10, 6))
sns.barplot(x='offensive_type', y='wage_eur', data=avg_wage_by_type, palette='viridis')
plt.title('Average Wage of Offensive Players by Type')
plt.ylabel('Average Wage (EUR)')
plt.xlabel('Player Type')
plt.xticks(rotation=15)
plt.show()


# In[30]:


# print average wages for offensive players by type
for index, row in avg_wage_by_type.iterrows():
    print(f"The average wage for {row['offensive_type']} is {row['wage_eur']:,.0f} EUR.")

