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

#-----------------------------  MODELING  ------------------------------#
# 1. FIND BEST MATCHES BASED ON CHECKBOX DATA
# reshape user input to be 2d to match our df
user_input_reshaped = user_input_vector.reshape(1, -1)

# calculate similarities
similarities = cosine_similarity(user_input_reshaped, df)

# sort occupations by similarity
similarities_sorted = np.argsort(similarities).flatten() # sort indexes from most to least similar

# 2. FIND CLOSEST MATCHES TO BEST MATCH
# use a wrapper function to use the same process for the intial plot and scatter plot clicks
def modeling_wrapper(best_match_id):
    ''' takes in the a specified number of points and a user input vector. returns a dataframe of 
    pca dimsensions with occupation details of n number of closest occupations to user input based 
    on KNN calculation. '''
    # find most similar based on distances using KNN
    # scale
    scaler = RobustScaler()
    df_scaled = scaler.fit_transform(df)

    # convert back to df
    df_scaled = pd.DataFrame(df_scaled, columns = df.columns, index = df.index)

    # select row
    best_match_row = df_scaled.iloc[best_match_id].values.reshape(1, -1) # convert series to array and reshape 

    # apply KNN
    knn = NearestNeighbors(n_neighbors = 30)
    knn.fit(df_scaled)
    distances, nearest_indexes = knn.kneighbors(best_match_row)

    # apply filtering
    most_similar_data = df.iloc[nearest_indexes.flatten()]

    # 3. APPLY PCA
    # dimensionality reduction
    pca = PCA(n_components = 2)
    reduced_data = pca.fit_transform(most_similar_data)

    # convert back to dataframe
    pca_df = pd.DataFrame(data = reduced_data, columns = ['PCA_1', 'PCA_2'], index = most_similar_data.index)

    # merge with occupation data to retrive titles and descriptions based on code index
    pca_df = pca_df.join(occupation_data)

    return pca_df

#-----------------------  WEBPAGE CONFIGURATION  -----------------------#
# 1. CREATE SCATTER PLOT    
# attempted to use plotly click data callback, ref https://dash.plotly.com/interactive-graphing
# only compatible with dash, not streamlit, use plotly_events instead   
def scatter_plot_generator(pca_df):
    fig = go.Figure()
    # base plot
    fig.add_trace(go.Scatter(x = pca_df['PCA_1'], 
                             y = pca_df['PCA_2'], 
                             mode = 'text+markers',
                             text = pca_df['Title'],
                             textposition = 'middle center',
                             textfont = dict(size = 16)))

    # remove plot features and specify plot height
    fig.update_layout(xaxis = dict(showline = False,
                                   zeroline = False,
                                   showgrid = False,
                                   showticklabels = False),
                      yaxis = dict(showline = False,
                                   zeroline = False,
                                   showgrid = False,
                                   showticklabels = False))
                      #hovermode = 'closest')
                      #height = 900) # applying here causes glitch, instead use override_height in plotly_events

    # ISSUES: 
    # removing hover info:
    #   - applying 'hovermode = False' within update layout disables click events
    #   - applying hovertemplate = 'none' to update_traces shows no info on pop up but not disabled
    #   - hoverinfo = 'skip' or 'none' to update traces does not seem to have any effect
    # only the marker can be clicked, see if we can change marker bounding box to match text

    return fig


# column config for plot and occupation desciption
col_list = st.columns([0.7, 0.3]) # set proportional width of cols

if 'selected_points' not in st.session_state:
    st.session_state.selected_points = None
selected_points = st.session_state.selected_points

if selected_points: # click event occured
    st.write('2nd iter')
    st.write(selected_points) # this is empty after a click, not sure why
    pca_df = modeling_wrapper(best_match_id = selected_points[0]['pointIndex']) # update based on id of clicked point (pull index from dict)
    with col_list[0]:
        selected_points = plotly_events(scatter_plot_generator(pca_df))# override_height = '700px') # this resizes but does not allow clicked data to be assigned to variable
        st.session_state.selected_points = selected_points # save to session state

    with col_list[1]:
        occupation = pca_df.iloc[0] # first row is most similar match
        st.subheader(f"{occupation['Title']}") # display occupation title
        st.write(f"{occupation['Description']}") # display occupation description

else: # first iteration of plot generation
    st.write('1st iter')
    pca_df = modeling_wrapper(best_match_id = similarities_sorted[0]) # generate dataframe

    # disaply plot
    with col_list[0]:
        selected_points = plotly_events(scatter_plot_generator(pca_df))# override_height = '700px') # this resizes but does not allow clicked data to be assigned to variable
        st.session_state.selected_points = selected_points # save to session state

    # display details
    with col_list[1]:
        occupation = pca_df.iloc[0] # first row is most similar match
        st.subheader(f"{occupation['Title']}") # display occupation title
        st.write(f"{occupation['Description']}") # display occupation description

    st.write(selected_points)