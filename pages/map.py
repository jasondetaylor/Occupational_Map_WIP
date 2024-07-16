import dash
from dash import html

dash.register_page(__name__, path = '/map')

layout = html.Div(html.H1('Map'))