# import libraries
import streamlit as st
import pandas as pd
import numpy as np

# import dataframes from jupyter notebook
# should add cache here as only need to import once
@st.cache_data
def load_dfs():
    df = pd.read_csv('df.csv', header = [0, 1], index_col = 0) # our cleaned data
    user_input_vars = pd.read_csv('user_input_vars.csv') # our user input variables
    return df, user_input_vars

df, user_input_vars = load_dfs()

# function to pick n random elements of each source from user_input_vars
def random_vars(df, n):
    '''takes in a dataframe and specified number of options, returns a dictionary containing df's of n number of metrics. 
    1 dict entry per metric group as governed by source col'''
    
    random_vars_dict = {} # empty dict to store indexes for each source
    for source in df['Source'].unique(): # iterate through each metric group via 'source' col
        filtered_df = df[df['Source'] == source] # filter df to contain only that source
        indexes = np.random.choice(len(filtered_df), size = n, replace = False) # generate n random indexes
        random_vars_dict[source] = filtered_df.iloc[indexes].drop('Source', axis = 1) # store rows
    return random_vars_dict

# select 10 random options for each source
if 'user_options' not in st.session_state:
    st.session_state.user_options = random_vars(user_input_vars, 10)

# STREAMLIT APP
st.title('Select options that resonate with you from the list below:')
col_list = st.columns(len(st.session_state.user_options))

for idx, source in enumerate(st.session_state.user_options.keys()): # iterate through dict
    with col_list[idx]: # place in column
        st.subheader(f'{source.capitalize()}:')
        for index, row in st.session_state.user_options[source].iterrows(): # iterate through df rows
            st.checkbox(label = row['Element Name'], key = row['Element ID']) # pull out label, use ID as key