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

#---------------------------  APP LAYOUT  ---------------------------#

app = Dash()

app.layout = html.Div([
              dcc.Checklist(id = 'user_input', options = user_options['knowledge']['Element Name']),
              html.Div(id = 'user_input_vector'), # show vector for now to check output
              html.Div(id = 'element_ids'),
              html.Button('Go!', id = 'go')
])

print(df.columns)
print(user_options['knowledge'])

#----------------------------  CALLBACK  ----------------------------#

@callback(
    Output(component_id='user_input_vector', component_property='children'),
    Input(component_id='go', component_property='n_clicks'),
    Input(component_id='user_input', component_property='value')
    # output and input above are arguments of the callback decorator
    # note component_id and component_property keywords are optional here
)
# the callback operator wraps this function below.
# whenever the input property changes, this function is called.
# the new input values are the argument of the function.
# dash then updates the value property of the output component with what is returned by this function
def update_output_div(n_clicks, selected):
    if n_clicks: # if button is clicked
        element_ids = user_options['knowledge'][user_options['knowledge']['Element Name'].isin(selected)]['Element ID'] # retrieve element id's of selected options
        vector_indexes = [df.columns.get_loc((element_id, 'IM')) for element_id in element_ids] # convert the id's to indexes of matching rows in df, look only at 'Importance' metric denoted 'IM'
        user_input_vector[vector_indexes] = 1 # set value to 1 at corresponding indexes to create vector for similarity analysis
        return str(user_input_vector)

    
#-------------------------------  RUN  -------------------------------#

if __name__ == '__main__':
    app.run(debug=True)



# PLAN
# create a go button (this will be your new input property for callback decorator) - done
# update user_inout_vector once user has finished checking boxes and clicks go (a new output property) - done
# expand checkboxes to 2 columns, 1 for knowledge and 1 for skills
# go to new page and display plot