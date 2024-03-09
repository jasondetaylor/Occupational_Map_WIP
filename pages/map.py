#-------------------------------  SETUP  -------------------------------#
# import libraries
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors

# access variables from main page
df = st.session_state.df
user_input_vector = st.session_state.user_input_vector

#-----------------------------  MODELING  ------------------------------#
# 1. FIND BEST MATCHES BASED ON CHECKBOX DATA
# reshape user input to be 2d to match our df
user_input_reshaped = user_input_vector.reshape(1, -1)

# calculate similarities
similarities = cosine_similarity(user_input_reshaped, df)

# find best match
similarities_sorted = np.argsort(similarities).flatten() # sort indexes from most to least similar
best_match_id = similarities_sorted[0]

# 2. FIND CLOSEST MATCHES TO BEST MATCH
n = 30 # number of points to display on plot

# option 1: based on similarity to user input vector
most_similar_indexes = similarities_sorted[:n]

# Option 2: based on distances using KNN
# scale
scaler = RobustScaler()
df_scaled = scaler.fit_transform(df)

# convert back to df
df_scaled = pd.DataFrame(df_scaled, columns = df.columns, index = df.index)

# select row
best_match_row = df_scaled.iloc[best_match_id].values.reshape(1, -1) # convert series to array and reshape 

# apply KNN
knn = NearestNeighbors(n_neighbors = n)
knn.fit(df_scaled)
distances, nearest_indexes = knn.kneighbors(best_match_row)

# apply filtering
#most_similar_data = df.iloc[most_similar_indexes.flatten()] # option 1
most_similar_data = df.iloc[nearest_indexes.flatten()] # option 2

# 3. APPLY PCA
# dimensionality reduction
pca = PCA(n_components = 2)
reduced_data = pca.fit_transform(most_similar_data)

# convert back to dataframe
pca_df = pd.DataFrame(data = reduced_data, columns = ['PCA_1', 'PCA_2'], index = most_similar_data.index)

# 4. MERGE WITH OCCUPATION DATA TO RETRIEVE TITLES AND DESCRIPTIONS
# read in occupation data
@st.cache_data
def load_occupation():
    occupation_data = pd.read_excel('db_28_2_excel/Occupation Data.xlsx')
    return occupation_data

occupation_data = load_occupation()
occupation_data.set_index('O*NET-SOC Code', inplace = True) # set code as index to match pca_df to make merging easier

# join df's based on codes
pca_df = pca_df.join(occupation_data)

#-----------------------  WEBPAGE CONFIGURATION  -----------------------#
# page title
st.write('Map')

# display scatter plot
scatter = sns.scatterplot(data = pca_df, x = 'PCA_1', y = 'PCA_2')
st.pyplot(scatter.figure)