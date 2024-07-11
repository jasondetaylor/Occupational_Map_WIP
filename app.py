from dash import Dash, html, dash_table
import pandas as pd

df = pd.read_csv('df.csv', header = [0, 1], index_col = 0) 

app = Dash()

app.layout = [
    html.Div(children = 'Hello World')]

if __name__ == '__main__':
    app.run(debug=True)