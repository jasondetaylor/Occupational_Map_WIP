# import libraries
import streamlit as st
import pandas as pd


df = pd.read_csv('df.csv', header = [0, 1, 2], index_col = 0)
st.dataframe(df)