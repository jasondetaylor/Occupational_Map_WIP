#-------------------------------  SETUP  -------------------------------#
# import libraries
import streamlit as st
import pandas as pd
import numpy as np
from streamlit_extras.switch_page_button import switch_page

# import dataframes exported from jupyter notebook
@st.cache_data
def load_dfs():
    df = pd.read_csv('df.csv', header = [0, 1], index_col = 0) # our cleaned data
    user_input_vars = pd.read_csv('user_input_vars.csv') # our user input variables
    return df, user_input_vars

# load in dfs
df, user_input_vars = load_dfs()
st.session_state.df = df # assign to session state to pass to map page

#-------------------------  DATA MANIPULATION  -------------------------#
# function to pick n random elements of each source from user_input_vars
def random_vars(df, n):
    '''takes in a dataframe and specified number of options, returns a 
    dictionary containing df's of n number of metrics. 1 dict entry per 
    metric group as governed by source col'''
    
    random_vars_dict = {} # empty dict to store indexes for each source
    for source in df['Source'].unique(): # iterate through each metric group via 'source' col
        filtered_df = df[df['Source'] == source] # filter df to contain only that source
        indexes = np.random.choice(len(filtered_df), size = n, replace = False) # generate n random indexes
        random_vars_dict[source] = filtered_df.iloc[indexes].drop('Source', axis = 1) # store rows
    return random_vars_dict

# select 10 random options for each source
if 'user_options' not in st.session_state:
    st.session_state.user_options = random_vars(user_input_vars, 10)

#-----------------------  WEBPAGE CONFIGURATION  -----------------------#
# page title
st.title('Select options that resonate with you from the list below:')

# initialize variables
col_list = st.columns(len(st.session_state.user_options)) # set up page cols
user_input_vector = np.zeros(df.shape[1]) # initialize user defined vector

# populate page columns with checkbox options
for idx, source in enumerate(st.session_state.user_options.keys()): # iterate through dict
    with col_list[idx]: # place in column
        st.subheader(f'{source.capitalize()}:')
        for index, row in st.session_state.user_options[source].iterrows(): # iterate through df rows
            selected = st.checkbox(label = row['Element Name'], key = row['Element ID']) # pull out label, use ID as key
            if selected:
                vector_index = df.columns.get_loc(row['Element ID']).start # multiheader indexing returns slice with starting value denoting 'IM' col index 
                user_input_vector[vector_index] = 1 # set value to 1 to create vector similarity analysis

# test to check that checkbox selection is overwriting values in user_inout_vector
st.write([i for i, value in enumerate(user_input_vector) if value == 1])

# assign to session state to pass varaible to map page
st.session_state.user_input_vector = user_input_vector 

# search button
search_button = st.button('Go')
if search_button:
    switch_page('map')

