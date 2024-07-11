#-------------------------------  SETUP  -------------------------------#
# import libraries
from dash import Dash, html, dash_table, dcc, callback, Input, Output
import pandas as pd
import numpy as np

# read in df's
df = pd.read_csv('df.csv', header = [0, 1], index_col = 0) # our cleaned data
user_input_vars = pd.read_csv('user_input_vars.csv') # our user input variables

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

# select 10 random options
user_options = random_vars(user_input_vars, 10)

# setup our user input vector to be populated by checklist selections
user_input_vector = np.zeros(df.shape[1])

print(user_options['knowledge']['Element Name'])

app = Dash()

app.layout = html.Div([
              dcc.Checklist(id = 'user_input', options = user_options['knowledge']['Element Name']),
              html.Div(id = 'element_id')
])

@callback(
    Output(component_id='element_id', component_property='children'),
    Input(component_id='user_input', component_property='value')
)

def update_output_div(input_value):
    return input_value


if __name__ == '__main__':
    app.run(debug=True)