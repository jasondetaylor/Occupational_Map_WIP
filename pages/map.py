#-------------------------------  SETUP  -------------------------------#
# import libraries
import dash
from dash import html, callback, Input, Output
import pandas as pd
import numpy as np

from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors

# re-read in df and occupation data df
df = pd.read_csv('df.csv', header = [0, 1], index_col = 0) # our cleaned data
occupation_data = pd.read_excel('db_28_2_excel/Occupation Data.xlsx')
occupation_data.set_index('O*NET-SOC Code', inplace = True) # set code as index to match pca_df to match previous df

#------------------------------  SCALING  -------------------------------#
# scale
scaler = RobustScaler()
df_scaled = scaler.fit_transform(df)

# convert back to df
df_scaled = pd.DataFrame(df_scaled, columns = df.columns, index = df.index)

#-----------------------------  FUNCTIONS  ------------------------------#


dash.register_page(__name__, path = '/map')

layout = html.Div([
    html.H1('Map'),
    html.Div(id = 'similarities')
])

@callback(
    Output('similarities', 'children'),
    Input('user_input_vector_store', 'data')  # Listen for data changes
)

def display_vector(user_input_vector):
    if user_input_vector is not None:
        # FIND BEST MATCHES BASED ON CHECKBOX DATA
        # calculate similarities
        user_input_vector = np.array(user_input_vector) # convert from list to array
        similarities = cosine_similarity(user_input_vector.reshape(1, -1), df) # with user input reshaped to 2d to match our df (note can use scaled or not here)
        # sort occupations by similarity
        similarities_idx_sorted = np.argsort(similarities).flatten() # sort indexes from most to least similar in array form
        return str(similarities_idx_sorted[0])
    return dash.no_update