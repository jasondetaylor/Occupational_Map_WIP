# import libraries
import streamlit as st
import pandas as pd

# import dataframes from jupyter notebook
# should add cache here as only need to import once
df = pd.read_csv('df.csv', header = [0, 1], index_col = 0) # our cleaned data
user_input_vars = pd.read_csv('user_input_vars.csv') # our user input variables

# function to select random options for a given source
@st.cache_data()
def select_random_options(df, n):
    '''takes in a dataframe and specified number of options, returns a dictionary containing lists of n number of metrics. 
    1 dict entry per metric group as governed by source col'''
    
    variable_lists = {}
    for var in df['Source'].unique():
        variable_lists[var] = df[df['Source'] == var].sample(n)['Element Name'].tolist()
    return variable_lists

# select 10 random options for each category
n = 10
user_options = select_random_options(user_input_vars, n = n)

# STREAMLIT APP
st.title('Select options that rsonate with you from the list below:')

# display user input options under separate columns
col_list = st.columns(len(user_options))  # dplit the screen into columns
selected_options = {}  # to store selected options

for idx, (source, options) in enumerate(user_options.items()):
    with col_list[idx]:
        st.subheader(f'{source.capitalize()}:')
        selected_options[source] = []
        for option in options:
            selected = st.checkbox(option)