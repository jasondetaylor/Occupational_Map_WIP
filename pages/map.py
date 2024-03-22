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

#-----------------------------  FUNCTIONS  ------------------------------#
# 1. FIND NEAREST NEIGHBORS TO OCCUPATION
# note: 'occupation' is either the highest cosine similarity score first 1st iteration, or click data for subsequent iterations
# use a wrapper function to use the same process for the intial plot and scatter plot clicks
def modeling_wrapper(best_match_idx):
    ''' takes in the a specified number of points and a best matched occupation index. returns a dataframe of 
    pca dimsensions with occupation details of n number of closest occupations to user input based 
    on KNN calculation. '''
    # 1A. FIND MOST SIMILAR BASED ON DISTANCES USING KNN
    # scale
    scaler = RobustScaler()
    df_scaled = scaler.fit_transform(df)

    # convert back to df
    df_scaled = pd.DataFrame(df_scaled, columns = df.columns, index = df.index)

    # select row
    best_match_row = df_scaled.iloc[best_match_idx].values.reshape(1, -1) # convert series to array and reshape 

    # apply KNN
    knn = NearestNeighbors(n_neighbors = 30)
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

    return pca_df


# 2. CREATE SCATTER PLOT    
# attempted to use plotly click data callback, ref https://dash.plotly.com/interactive-graphing
# only compatible with dash, not streamlit, use plotly_events instead   
def scatter_plot_generator(pca_df):
    fig = go.Figure()
    # base plot
    fig.add_trace(go.Scatter(x = pca_df['PCA_1'], 
                             y = pca_df['PCA_2'], 
                             mode = 'text+markers',
                             text = pca_df['Title'],
                             textposition = 'top center',
                             textfont = dict(size = 16, color = '#4DBEEE')))

    # remove plot features and specify plot height
    fig.update_layout(xaxis = dict(showline = False,
                                   zeroline = False,
                                   showgrid = False,
                                   showticklabels = False),
                      yaxis = dict(showline = False,
                                   zeroline = False,
                                   showgrid = False,
                                   showticklabels = False),
                      paper_bgcolor = 'rgba(0,0,0,0)',
                      plot_bgcolor = 'rgba(0,0,0,0)')
                      #hovermode = 'closest')
                      #height = 900) # applying here causes glitch, instead use override_height in plotly_events

    # ISSUES: 
    # removing hover info:
    #   - applying 'hovermode = False' within update layout disables click events
    #   - applying hovertemplate = 'none' to update_traces shows no info on pop up but not disabled
    #   - hoverinfo = 'skip' or 'none' to update traces does not seem to have any effect
    # only the marker can be clicked, see if we can cuse a clickable bounding box to match text length

    return fig


# 3. POPULATE WEBPAGE
def page_layout(best_match_idx):
    pca_df = modeling_wrapper(best_match_idx = best_match_idx) 
    with col_list[0]:
        selected_points = plotly_events(scatter_plot_generator(pca_df))# override_height = '700px') # this resizes but does not allow clicked data to be assigned to variable

    with col_list[1]:
        occupation = pca_df.iloc[0] # first row is most similar match
        st.subheader(f"{occupation['Title']}") # display occupation title
        st.write(f"{occupation['Description']}") # display occupation description

    return pca_df, selected_points
    
#-----------------------------  MODELING AND DISPLAY  ------------------------------#
# column config for plot and occupation desciption
col_list = st.columns([0.7, 0.3]) # set proportional width of cols

# initialize session state for selected_points
if 'selected_points' not in st.session_state:
    st.session_state.selected_points = None
selected_points = st.session_state.selected_points

if selected_points: # click event occured
    st.write('2nd iter')
    # generated df and plot
    pca_df, selected_points = page_layout(best_match_idx = selected_points[0]['pointIndex']) # update based on id of clicked point (pull index from dict)
    st.session_state.selected_points = selected_points # save to session state

else: # first iteration of plot generation
    st.write('1st iter')
    # FIND BEST MATCHES BASED ON CHECKBOX DATA
    # calculate similarities
    similarities = cosine_similarity(user_input_vector.reshape(1, -1), df) # reshape user input to be 2d to match our df
    # sort occupations by similarity
    similarities_idx_sorted = np.argsort(similarities).flatten() # sort indexes from most to least similar
    # generated df and plot
    pca_df, selected_points = page_layout(best_match_idx = similarities_idx_sorted[0]) # based on most similar match
    st.session_state.selected_points = selected_points # save to session state


st.write(selected_points)
st.write(pca_df)