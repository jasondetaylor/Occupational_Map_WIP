#-------------------------------  SETUP  -------------------------------#
# import libraries
import dash
from dash import html, callback, dcc, Input, Output
import pandas as pd
import numpy as np
import plotly.express as px
import json

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
# FIND BEST MATCHES BASED ON CHECKBOX DATA
def find_similarities(user_input_vector, df):
    # calculate similarities
    user_input_vector = np.array(user_input_vector) # convert from list to array
    similarities = cosine_similarity(user_input_vector.reshape(1, -1), df) # with user input reshaped to 2d to match our df (note can use scaled or not here)
    # sort occupations by similarity
    similarities_idx_sorted = np.argsort(similarities).flatten() # sort indexes from most to least similar in array form
    best_match_code = df.iloc[similarities_idx_sorted[0]].name # retrieve best match code
    return best_match_code# output best match code


# FIND NEAREST NEIGHBORS TO OCCUPATION
# note: 'occupation' is either the highest cosine similarity score first 1st iteration, or click data for subsequent iterations
# use a wrapper function to use the same process for the intial plot and scatter plot clicks
def modeling_wrapper(code, df_scaled):
    ''' takes in the a specified number of points and a best matched occupation index. returns a dataframe of 
    pca dimsensions with occupation details of n number of closest occupations to user input based 
    on KNN calculation. '''
    # 1A. FIND MOST SIMILAR BASED ON DISTANCES USING KNN
    # select row
    # here is the issue, we are taking the index from our plot and applying it to df_scaled, try and get code from plot instead
    #best_match_row = df_scaled.iloc[best_match_idx].values.reshape(1, -1) # convert series to array and reshape 
    best_match_row = df_scaled.loc[code].values.reshape(1, -1) # convert series to array and reshape 

    # apply KNN
    knn = NearestNeighbors(n_neighbors = 20)
    knn.fit(df_scaled)
    distances, nearest_indexes = knn.kneighbors(best_match_row)

    # apply filtering
    most_similar_data = df.iloc[nearest_indexes.flatten()]

    # 1B. APPLY PCA
    # dimensionality reduction
    pca = PCA(n_components = 2)
    reduced_data = pca.fit_transform(most_similar_data)

    # convert back to dataframe
    pca_df = pd.DataFrame(data = reduced_data, columns = ['PCA_1', 'PCA_2'], index = most_similar_data.index)

    # 1C. CREATE DATAFRAME TO PASS TO PLOT
    # merge with occupation data to retrive titles and descriptions based on code index
    pca_df = pca_df.join(occupation_data)

    return pca_df#, code

#-----------------------------  PAGE SETUP  ------------------------------#

dash.register_page(__name__, path = '/map')

layout = html.Div([
    html.H1('Map'),
    html.Div(id = 'code'),
    dcc.Graph(id = 'map'),
    html.Div(id = 'title'),
    html.Div(id = 'description'),
    html.Div(id = 'click-data'),
    dcc.Store(id = 'pca_data') # store data to pass to click data callback
])

@callback(
    Output('code', 'children'),
    Output('pca_data', 'data'),
    Output('map', 'figure'),
    Output('title', 'children'),
    Output('description', 'children'),
    Input('user_input_vector_store', 'data')  # Listen for data changes
)

def display_vector(user_input_vector):
    if user_input_vector is not None:
        code = find_similarities(user_input_vector, df)
        pca_df = modeling_wrapper(code, df_scaled)
        fig = px.scatter(pca_df, x = 'PCA_1', y = 'PCA_2')
        fig.update_layout(clickmode='event+select')
        title = occupation_data.loc[code]['Title']
        description = occupation_data.loc[code]['Description']
        return code, pca_df.to_json(), fig, title, description # note pca_df converted to json to pass to store
    return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update

@callback(
    Output('click-data', 'children'),
    Input('map', 'clickData'),
    Input('pca_data', 'data') # retrieve pca data
    )
def display_click_data(clickData, pca_data):
    if clickData is not None:
        click_dict = json.loads(json.dumps(clickData, indent=2)) # convert from str back to dict using json.loads
        pca_df = pd.read_json(pca_data) # convert back to df
        code = pca_df.index[click_dict['points'][0]['pointIndex']] # reteieve code using point index
        return code
    return dash.no_update