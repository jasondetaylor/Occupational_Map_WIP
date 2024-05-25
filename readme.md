# Interactive Occupational Map
## Motivation
Choosing a career path is one of the most significant decisions individuals make in their lives, often influencing their personal and professional fulfillment. This project aims to provide valuable insights and guidance to individuals, particularly students and job seekers, as they navigate the complex landscape of career choices.

By leveraging dimensionality reduction, this project explores the similarity between different occupations based on a wide range of metrics. These metrics include skills, knowledge, experience, and other factors obtained from the comprehensive O*NET dataset.

The aim of this project is to offer an interactive platform where users can visualize the relationships between occupations, identify clusters of similar professions, and gain a deeper understanding of potential career paths. Whether you're a high school student exploring future educational opportunities, a college graduate contemplating career options, or a seasoned professional considering a career change, this project aims to empower you with valuable insights to make informed decisions about your professional journey.

## Project Issues
Unfortunately, I am unable to achieve the standard of UI that I would deem acceptable given my current tools and abilites. Multiple issues arising with the map/plot have forced me to reconsider the current approach to this project, namely: 
- Unable to adjust plot height
- Unable to fit text inside plot borders without drastically narrowing the usable plot area
- Unable to make text clickable, rather than the data point
- Unable to control overlap of text with other data points

The culmination of these issues that I am unable to overcome without delving into potntially a very deep and fruitless effort has made me recognize the need for a different approach. Perhaps this project is better suited to a more customizable platform such as Dash or HTML in order to acheive the level of UI that I beleieve is neccessary for this project to be a success. Usability should be this product's greatest assest, without it, this tool will not reach those who need it.

## Conclusion
This arm of the project is terminated with the aim of continuing or restarting by a different means that allows improved UI.

## Appendices
### Data Source
https://www.onetcenter.org/database.html#all-files
### File List
**pages/map.py** - The streamlit script governing the map generation and regeneration after clicks.
**Occupational Map.ipynb** - Initial Jupyter notebook used to hash out the backbone of this project used to create the plot. Namely data manipulation, dimensionality reduction and KNN/cosine similarity.
**df.csv** - Tabular data retrived from data source given above.
**environment.yml** - use prompt "conda env create -f environment.yml" to recreate this env on your machine to run this project.
**streamlit_script.py** - The streamlit script for the checkbox landing page used to determine the inital best match
**user_input_vars.csv** - The variables used to create the options on the landing page.