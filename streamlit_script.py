import streamlit as st
import pandas as pd

df = pd.read_csv('df.csv')
st.dataframe(df)