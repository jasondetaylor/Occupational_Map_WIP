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

# note: should I be accessing df using dcc.Store from landing page instead of re-loading here?

#------------------------------  SCALING  -------------------------------#

# Choose robust scaler to reduce the effect that outliers may have on our PCA vectors
scaler = RobustScaler() # instantiate
df_scaled = scaler.fit_transform(df) # fit
df_scaled = pd.DataFrame(df_scaled, columns = df.columns, index = df.index) # convert back to df

#-----------------------------  FUNCTIONS  ------------------------------#

# APPLY KNN TO FIND NEAREST NEIGHBORS
def apply_KNN(row, n_neighbors, df):
    knn = NearestNeighbors(n_neighbors = n_neighbors)
    knn.fit(df)
    distances, nearest_indexes = knn.kneighbors(row)
    most_similar_data = df.iloc[nearest_indexes.flatten()]
    return most_similar_data

# USE CHECKBOX DATA TO FIND BEST MATCH OCCUPATION
def find_best_match(selected_codes, df_scaled):
    df_filtered = df_scaled[selected_codes] # filter df down to only columns matching user selection
    user_input = np.array(df_filtered.max(axis = 0)).reshape(1, -1) # create data point using max value of each col
    best_match_row = apply_KNN(row = user_input, n_neighbors = 1, df = df_filtered) # apply KNN to find closest neighbor
    best_match_code = best_match_row.index # retrieve code
    return best_match_code


# FIND NEAREST NEIGHBORS TO OCCUPATION
# note: 'occupation' is either the highest cosine similarity score first 1st iteration, or click data for subsequent iterations
# use a wrapper function to use the same process for the intial plot and scatter plot clicks
def modeling_wrapper(code, df_scaled):
    ''' takes in the a specified number of points and a best matched occupation index. returns a dataframe of 
    pca dimsensions with occupation details of n number of closest occupations to user input based 
    on KNN calculation. '''
    # 1A. FIND MOST SIMILAR BASED ON DISTANCES USING KNN
    # select row
    best_match_row = df_scaled.loc[code].values.reshape(1, -1) # convert series to array and reshape 
    # apply KNN
    most_similar_data = apply_KNN(row = best_match_row, n_neighbors = 20, df = df_scaled)

    # 1B. APPLY PCA
    # dimensionality reduction
    pca = PCA(n_components = 2)
    reduced_data = pca.fit_transform(most_similar_data)

    # convert back to dataframe
    pca_df = pd.DataFrame(data = reduced_data, columns = ['PCA_1', 'PCA_2'], index = most_similar_data.index)

    # 1C. CREATE DATAFRAME TO PASS TO PLOT
    # merge with occupation data to retrive titles and descriptions based on code index
    pca_df = pca_df.join(occupation_data)

    return pca_df

# GENERATE SCATTER PLOT
def map_display(pca_df, code):
    fig = px.scatter(pca_df, x = 'PCA_1', y = 'PCA_2', text = 'Title')
    fig.update_layout(clickmode = 'event+select',
                      xaxis_title = '', 
                      yaxis_title = '',
                      xaxis = dict(showticklabels = False),
                      yaxis = dict(showticklabels = False))
    title = pca_df.loc[code]['Title']
    description = pca_df.loc[code]['Description']

    return fig, title, description


#-----------------------------  PAGE SETUP  ------------------------------#

dash.register_page(__name__, path = '/map')

# edit this parameter to change relative plot width (%)
plot_width = 80

layout = html.Div([
    html.Div([
        html.Div(
            dcc.Graph(id = 'map'),
            style={'width': f'{plot_width}%', 'display': 'inline-block', 'verticalAlign': 'top'}),
        html.Div([
            html.H3(id = 'title'),
            html.Div(id = 'description'),
        ], style={'width': f'{100 - plot_width}%', 'display': 'inline-block', 'verticalAlign': 'top'})
        ], style={'display': 'flex'}),
    dcc.Store(id = 'pca_data'), # store previous pca_df
])

@callback(
    Output('pca_data', 'data'),
    Output('map', 'figure'),
    Output('title', 'children'),
    Output('description', 'children'),
    Input('selected_codes', 'data'),  # Listen for data changes
    Input('map', 'clickData'),
    Input('pca_data', 'data') # retrieve pca data
)

def display_map(selected_codes, clickData, pca_data):

    if selected_codes is None: # do nothing until initial input is recieved
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update

    elif clickData is None: # first iteration, checklist generated
        code = find_best_match(selected_codes, df_scaled)
        pca_df = modeling_wrapper(code, df_scaled)
        fig, title, description = map_display(pca_df, code)
        return pca_df.to_json(), fig, title, description

    elif clickData is not None: # 2nd -> nth iteration, uses click data
        click_dict = json.loads(json.dumps(clickData, indent=2)) # convert from str back to dict using json.loads
        pca_df = pd.read_json(pca_data)
        new_code = pca_df.index[click_dict['points'][0]['pointIndex']] # retrieve code using point index
        new_pca_df = modeling_wrapper(new_code, df_scaled)
        new_fig, new_title, new_description = map_display(new_pca_df, new_code)
        return new_pca_df.to_json(), new_fig, new_title, new_description
    
    # To Do List
    # General:
    #   - edit theme
    # Map:
    #   - make fullscreen
    #   - edit axes to not cut off text data
    #   - disallow text overlap
    #   - make text clickabe, not just data point
    #   - add a go back button
    # Landing Page:
    #   - maybe: append selected options to a list, replace selected with new option, add refresh button to regenerate lists
