#-------------------------------  SETUP  -------------------------------#
# import libraries
import dash
from dash import Dash, html, dcc, callback, Input, Output, State
import dash_bootstrap_components as dbc
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

app = Dash(__name__, use_pages=True, external_stylesheets=[dbc.themes.MINTY])

column_width = 100 / len(user_options.keys()) # calculate width percentage to pass to style in checklist to define number of columns

checklist_component = [
    html.Div([
        html.H3(source),
        dcc.Checklist(
            id = f'user_input_{source}', 
            options = [{'label': row['Element Name'], 'value': row['Element ID']} for _, row in df.iterrows()] # use name as label, id as value
        )
    ], style={'width': f'{column_width}%', 'display': 'inline-block'}) # dynamic col setup, 1 col for each source
    for i, (source, df) in enumerate(user_options.items())
]

app.layout = html.Div([
              dcc.Location(id = 'url', refresh = False),  # Include dcc.Location for routing
              dcc.Store(id = 'selected_codes'),  # Store for the checlist selection to pass to map page
              html.H2('Select options that resonate with you from the list below:'),
              html.Plaintext('The more options selected, the more accurate the intial match'),
              html.Div(checklist_component),
              html.Button('Go!', id = 'go'),
              dash.page_container
])

#----------------------------  CALLBACK  ----------------------------#

# Generate a list of Input and State objects for the callback
checklist_component_ids = [comp.children[1].id for comp in checklist_component] # extract IDs from checklist_component
input_list = [State(checklist_id, 'value') for checklist_id in checklist_component_ids] # generate list State objects for the callback
input_list.append(Input('go', "n_clicks")) # add in go button input

@callback(
    Output('selected_codes', 'data'),
    Output('url', 'pathname'),
    *input_list # unpack list of inputs
)
def update_output_div(selected1, selected2, n_clicks):
    if n_clicks: # if button is clicked
        selected1, selected2 = None_type_to_list(selected1), None_type_to_list(selected2) # convert to empty list if no options from that list are checked
        selected = selected1 + selected2
        return selected, '/map'
    return dash.no_update, dash.no_update # do nothing if button is not clicked

#-------------------------------  RUN  -------------------------------#

if __name__ == '__main__':
    app.run(debug=True)