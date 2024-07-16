#-------------------------------  SETUP  -------------------------------#
# import libraries
import dash
from dash import Dash, html, dcc, callback, Input, Output, State
import pandas as pd
import numpy as np

# read in df's
df = pd.read_csv('df.csv', header = [0, 1], index_col = 0) # our cleaned data
user_input_vars = pd.read_csv('user_input_vars.csv') # our user input variables

#-----------------------------  FUNCTIONS  -----------------------------#
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


def None_type_to_list(list):
    ''' Converts type None to empty list'''

    list = list if list is not None else []
    
    return list

#-------------------------  DATA MANIPULATION  -------------------------#
# select 10 random options
user_options = random_vars(user_input_vars, 10)

# setup our user input vector to be populated by checklist selections
user_input_vector = np.zeros(df.shape[1])

#---------------------------  APP LAYOUT  ---------------------------#

app = Dash(__name__, use_pages=True)

checklist_component_ids = [f'user_input_{source}' for source in user_options.keys()]

checklist_component = [
    html.Div([
        dcc.Checklist(
            id = checklist_component_ids[i], 
            options = [{'label': row['Element Name'], 'value': row['Element ID']} for _, row in df.iterrows()] # use name as label, id as value
        )
    ])
    for i, df in enumerate(user_options.values())
]

app.layout = html.Div([
              dcc.Location(id='url', refresh=False),  # Include dcc.Location for routing
              html.Div(checklist_component),
              html.Div(id = 'user_input_vector'), # show vector for now to check output
              html.Div(id = 'element_ids'),
              html.Button('Go!', id = 'go'),
              dash.page_container
])

#print(df.columns)
#print(user_options['knowledge'])

#----------------------------  CALLBACK  ----------------------------#

# Generate a list of Input and State objects for the callback
input_list = [State(checklist_id, 'value') for checklist_id in checklist_component_ids]
input_list.append(Input('go', "n_clicks"))

@callback(
    Output(component_id='user_input_vector', component_property='children'),
    Output('url', 'pathname'),
    *input_list # unpack list of inputs
    # output and input above are arguments of the callback decorator
    # note component_id and component_property keywords are optional here
)
# the callback operator wraps this function below.
# whenever the input property changes, this function is called.
# the new input values are the argument of the function.
# dash then updates the value property of the output component with what is returned by this function
def update_output_div(selected1, selected2, n_clicks):
    if n_clicks: # if button is clicked
        selected1, selected2 = None_type_to_list(selected1), None_type_to_list(selected2) # convert to empty list if no options from that list are checked
        selected = selected1 + selected2
        vector_indexes = [df.columns.get_loc((id, 'IM')) for id in selected] # convert the id's to indexes of matching rows in df, look only at 'Importance' metric denoted 'IM'
        user_input_vector[vector_indexes] = 1 # set value to 1 at corresponding indexes to create vector for similarity analysis
        return str(user_input_vector), '/map'
    return dash.no_update # do nothing if button is not clicked

    
#-------------------------------  RUN  -------------------------------#

if __name__ == '__main__':
    app.run(debug=True)