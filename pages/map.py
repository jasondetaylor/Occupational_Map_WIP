import dash
from dash import html, callback, Input, Output

dash.register_page(__name__, path = '/map')

layout = html.Div([
    html.H1('Map'),
    html.Div(id = 'vector')
])

@callback(
    Output('vector', 'children'),
    Input('user_input_vector_store', 'data')  # Listen for data changes
)

def display_vector(user_input_vector):
    if user_input_vector is not None:
        return str(user_input_vector)
    return 'No vector data available.'