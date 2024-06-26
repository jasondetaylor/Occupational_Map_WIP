#-------------------------------  SETUP  -------------------------------#
# import libraries
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import plotly.express as px
from streamlit_plotly_events import plotly_events

from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors

# use full page width
st.set_page_config(layout = "wide")

# access variables from main page
df = st.session_state.df
user_input_vector = st.session_state.user_input_vector

# read in occupation data
@st.cache_data
def load_occupation():
    occupation_data = pd.read_excel('db_28_2_excel/Occupation Data.xlsx')
    return occupation_data

occupation_data = load_occupation()
occupation_data.set_index('O*NET-SOC Code', inplace = True) # set code as index to match pca_df to match previous df

#------------------------------  SCALING  -------------------------------#
# scale
scaler = RobustScaler()
df_scaled = scaler.fit_transform(df)

# convert back to df
df_scaled = pd.DataFrame(df_scaled, columns = df.columns, index = df.index)

#-----------------------------  FUNCTIONS  ------------------------------#
# 1. FIND NEAREST NEIGHBORS TO OCCUPATION
# note: 'occupation' is either the highest cosine similarity score first 1st iteration, or click data for subsequent iterations
# use a wrapper function to use the same process for the intial plot and scatter plot clicks
def modeling_wrapper(code, df_scaled = df_scaled):
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

    return pca_df, code

# 2. CREATE SCATTER PLOT    
# attempted to use plotly click data callback, ref https://dash.plotly.com/interactive-graphing
# only compatible with dash, not streamlit, use plotly_events instead   
def scatter_plot_generator(pca_df):
    fig = go.Figure()
    # base plot
    x_data = pca_df['PCA_1']
    y_data = pca_df['PCA_2']
    
    fig.add_trace(go.Scatter(x = x_data, 
                             y = y_data, 
                             mode = 'text+markers',
                             text = pca_df['Title'],
                             textposition = 'top center',
                             textfont = dict(size = 16, color = '#4DBEEE'))) # 'MATLAB blue'

    # remove plot features and specify plot height
    scale_factor = 8
    fig.update_layout(xaxis = dict(showline = False,
                                   zeroline = False,
                                   showgrid = False,
                                   showticklabels = False,
                                   range = [min(x_data) * scale_factor, max(x_data) * scale_factor]), # manually adjust x axis range to fit text within vounds of plot
                      yaxis = dict(showline = False,
                                   zeroline = False,
                                   showgrid = False,
                                   showticklabels = False),
                      hovermode = 'closest',
                      paper_bgcolor = 'rgba(0,0,0,0)',
                      plot_bgcolor = 'rgba(0,0,0,0)')
                      #height = 900) # applying here causes glitch

    # ISSUES: 
    # cannot adjust plot height without causing glitvh or mishandling of click events
    # removing hover info:
    #   - applying 'hovermode = False' within update layout disables click events
    #   - applying hovertemplate = 'none' to update_traces shows no info on pop up but not disabled
    #   - hoverinfo = 'skip' or 'none' to update traces does not seem to have any effect
    # only the marker can be clicked, see if we can cuse a clickable bounding box to match text length

    return fig


# 3. POPULATE WEBPAGE
def page_layout(code):
    pca_df, code = modeling_wrapper(code)

    with col_list[0]:
        selected_points = plotly_events(scatter_plot_generator(pca_df))# override_height = '700px') # this resizes but does not allow clicked data to be assigned to variable consitently

    with col_list[1]:
        occupation = pca_df.iloc[0] # first row is most similar match            
        st.subheader(f"{occupation['Title']}") # display occupation title
        st.write(f"{occupation['Description']}") # display occupation description

    # only recompute code on second iteration (after plot has been clicked)
    if selected_points:
        code = pca_df.index[selected_points[0]['pointIndex']] # recompute code

    return pca_df, selected_points, code

#-----------------------------  MODELING AND DISPLAY  ------------------------------#
# column config for plot and occupation description
col_list = st.columns([0.7, 0.3]) # set proportional width of cols

# initialize session state for selected_points
if 'selected_points' not in st.session_state:
    st.session_state.selected_points = []
# pull selected_points from session state
selected_points = st.session_state.selected_points

# initialize session state for code
if 'code' not in st.session_state:
    st.session_state.code = None

# pull code from session state
code = st.session_state.code

# notes on rerun function:
# when a the plot is clicked, plotly_events popluates the selected_points dict with the click data
# however this click data is an output and is not yet used as input to generate a new plot
# to get around this we will define a rerun function to allow us to skip the instance where the plot is clicked but data is not yet used so every click regenerates the plot
def rerun():
    if selected_points: # only rerun after plot has been clicked, after new plot is generated selected points is empty
        st.rerun() # skip this iteration to regenerate the plot

# use cache decorator to store selected_points in the session state and retain data across reruns
@st.cache_data
def selected_points_store(selected_points):
    st.session_state.selected_points = selected_points
selected_points = st.session_state.selected_points

if selected_points:
    # FIND BEST MATCHES BASED ON CLICK DATA
    # generated df and plot
    pca_df, selected_points, code = page_layout(code) # based on id of clicked point (pull index from dict)
    selected_points_store(selected_points) # save to session state with cache active
    st.session_state.code = code # save to session state
    rerun() # envoke a rerun

else: # first iteration of plot generation
    # FIND BEST MATCHES BASED ON CHECKBOX DATA
    # calculate similarities
    similarities = cosine_similarity(user_input_vector.reshape(1, -1), df) # with user input reshaped to 2d to match our df (note can use scaled or not here)
    # sort occupations by similarity
    similarities_idx_sorted = np.argsort(similarities).flatten() # sort indexes from most to least similar in array form
    # generated df and plot
    pca_df, selected_points, code = page_layout(code = df.index[similarities_idx_sorted[0]]) # based on most similar match
    selected_points_store(selected_points)
    st.session_state.code = code # save to session state
    rerun() # envoke a rerun