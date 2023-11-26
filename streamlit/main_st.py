import streamlit as st
import folium 
from streamlit_folium import folium_static
import pandas as pd
import plotly.express as px
import os
import plotly.graph_objects as go


######################### DATA FROM MODELS (EDIT) ############################################
current_directory = os.path.dirname(os.path.realpath(__file__))

## ------------- WELLS PRED ------------
#### AEK201
AEK201_path = os.path.join(current_directory, 'data', 'model_predictions_AEK201.pkl')
with open(AEK201_path, 'rb') as AEK201_data:
    AEK201 = pd.read_pickle(AEK201_data)

####AFL259
AFL259_path = os.path.join(current_directory, 'data', 'model_predictions_AFL259.pkl')
with open(AFL259_path, 'rb') as AFL259_data:
    AFL259 = pd.read_pickle(AFL259_data)

####APK309
APK309_path = os.path.join(current_directory, 'data', 'model_predictions_APK309.pkl')
with open(APK309_path, 'rb') as APK309_data:
    APK309 = pd.read_pickle(APK309_data)

####APK310
APK310_path = os.path.join(current_directory, 'data', 'model_predictions_APK310.pkl')
with open(APK310_path, 'rb') as APK310_data:
    APK310 = pd.read_pickle(APK310_data)

## ------------- WELLS SCORES ------------

#### SCORE AEK201 
AEK201_score_path= os.path.join(current_directory, 'data', 'model_scores_AEK201.pkl')
with open(AEK201_score_path, 'rb') as AEK201_score:
    AEK201_scores = pd.read_pickle(AEK201_score)

#### SCORE AFL259 
AFL259_score_path= os.path.join(current_directory, 'data', 'model_scores_AFL259.pkl')
with open(AFL259_score_path, 'rb') as AFL259_score:
    AFL259_scores = pd.read_pickle(AFL259_score)

#### SCORE APK309 
APK309_score_path= os.path.join(current_directory, 'data', 'model_scores_APK309.pkl')
with open(APK309_score_path, 'rb') as APK309_score:
    APK309_scores = pd.read_pickle(APK309_score)

#### SCORE APK310
APK310_score_path= os.path.join(current_directory, 'data', 'model_scores_APK310.pkl')
with open(APK310_score_path, 'rb') as APK310_score:
    APK310_scores = pd.read_pickle(APK310_score)


######################### PLOTTING ONLY ############################################

def plot_multitraces(df):
    selected_traces = st.multiselect('Select models to display:', ['Actual', 'Baseline', 'LinReg', 'CNN', 'LSTM'])

    fig = go.Figure()

    if 'Actual' in selected_traces:
        fig.add_trace(go.Scatter(x=df['date'], y=df['Actual'], name='Actual'))

    if 'Baseline' in selected_traces:
        fig.add_trace(go.Scatter(x=df['date'], y=df['Baseline'], name='Baseline'))

    if 'LinReg' in selected_traces:
        fig.add_trace(go.Scatter(x=df['date'], y=df['Linear Reg'], name='LinReg'))

    if 'CNN' in selected_traces:
        fig.add_trace(go.Scatter(x=df['date'], y=df['CNN'], name='CNN'))

    if 'LSTM' in selected_traces:
        fig.add_trace(go.Scatter(x=df['date'], y=df['LSTM'], name='LSTM'))

    fig.update_xaxes(rangeslider_visible=True,
                     rangeselector=dict(
                         buttons=list([
                             dict(count=7,
                                  label='1w',
                                  step='day',
                                  stepmode='backward'),
                             dict(count=1,
                                  label='1m',
                                  step='month',
                                  stepmode='backward'),
                             dict(count=3,
                                  label='3m',
                                  step='month',
                                  stepmode='backward'),
                             dict(count=6,
                                  label='6m',
                                  step='month',
                                  stepmode='backward'),
                             dict(count=1,
                                  label='1y',
                                  step='year',
                                  stepmode='backward'),
                             dict(step='all')
                         ])))

    fig.add_trace(go.Scatter(x=[], y=[], mode='lines', line=dict(color='red', dash='dash'), name='Vertical Line'))
    fig.update_layout(hovermode='x unified')

    def update_vertical_line(trace, points, selector):
        if points.point_inds:
            selected_date = df['date'][points.point_inds[0]]
            fig.update_traces(x=[selected_date, selected_date],
                              y=[0, max(df['Actual'].max(), df['Baseline'].max())])

    fig.data[-1].on_hover(update_vertical_line)

    fig.update_layout(margin=dict(l=0, r=0, b=0, t=50))

    st.plotly_chart(fig)


current_directory = os.path.dirname(os.path.realpath(__file__))


#####################################################################


# Main Title/Header
st.title("Groundwater Level Forecast")
st.write("This project was done as part of the [Fall 2023 Erdös Institute Datascience Bootcamp](https://www.erdosinstitute.org/). \
    To learn more about the project visit the [github repository](https://github.com/meresar/erdos_groundwater).")

# Map (Fixed)
def init_map(center=[47.655280, -117.240325], zoom_start=10, tiles="cartodbpositron"):
    return folium.Map(location=center, zoom_start=zoom_start, tiles=tiles)

# Sidebar 
st.sidebar.header("Well Selection")
st.sidebar.write("To see exact well locations, hover over the droppins in the map.")
selected_well = st.sidebar.selectbox("Select your well:", ['','AEK201', 'AFL259', 'APK309', 'APK310'])
with st.sidebar.expander("Well Information", expanded=True):  # Set expanded to True if you want it to be open by default
    st.write("""
        - AEK201: East Central Spokane
        - AFL259: Otis Orchards-East Farms
        - AEK309: Plantes Ferry Park 
        - AEK310: Wellesley Ave, Spokane Valley
        """)


# Plots

## Home Page select ""
if selected_well == '':
    ## Map (All Red)
    m = init_map()
    folium.Marker(location=[47.6619,-117.3655],
                 tooltip = 'AEK201',
                 icon = folium.Icon(color='red', icon="bookmark", prefix='fa')).add_to(m)
    folium.Marker(location=[47.6946,-117.1011],
                 tooltip = 'AFL259',
                 icon = folium.Icon(color='red', icon="bookmark", prefix='fa')).add_to(m)
    folium.Marker(location=[47.698,-117.2395],
                 tooltip = 'APK309',
                 icon = folium.Icon(color='red', icon="bookmark", prefix='fa')).add_to(m)
    folium.Marker(location=[47.7007,-117.1954],
                 tooltip = 'APK310',
                 icon = folium.Icon(color='red', icon="bookmark", prefix='fa')).add_to(m)
    folium_static(m, height = 300)

    st.header('About the Project')
    st.write('Groundwater is a critical source of water for human survival. A significant percentage of both drinking \
        and crop irrigation water is drawn from groundwater sources through wells. \
        In the US, overuse of groundwater could have major implications for the future. \
        Machine learning, deep learning, and time series techniques have shown promise for helping understand and address these problems. \
        In this project we implemented Convolutional Neural Network (CNN) and Long Short-Term Memory (LSTM) to predict Groundwater Levels \
        (GWL) in selected wells located in Columbia Plateau basaltic-rock and basin-fill aquifers: one of the principal aquifers of the U.S \
        using historical well, weather and surface data.')


## AEK201 
if selected_well == "AEK201":
    ## Map (Selected blue)
    m = init_map()
    folium.Marker(location=[47.6619,-117.3655],
                 tooltip = 'AEK201',
                 icon = folium.Icon(color='blue', icon="bookmark", prefix='fa')).add_to(m)
    folium.Marker(location=[47.6946,-117.1011],
                 tooltip = 'AFL259',
                 icon = folium.Icon(color='red', icon="bookmark", prefix='fa')).add_to(m)
    folium.Marker(location=[47.698,-117.2395],
                 tooltip = 'APK309',
                 icon = folium.Icon(color='red', icon="bookmark", prefix='fa')).add_to(m)
    folium.Marker(location=[47.7007,-117.1954],
                 tooltip = 'APK310',
                 icon = folium.Icon(color='red', icon="bookmark", prefix='fa')).add_to(m)
    folium_static(m, height = 300)

    st.header('About the Well')
    st.write('AEK201 is a groundwater monitoring well in Spokane County, Washington (Latitude/Longitude: 47.6619, -117.3655) \
        at an elevation of 1,940 ft within the Columbia Plateau basin-fill aquifers. The well is 105 ft deep and the lithography \
        varies at different depths. The top 42 ft of the well consists of gravel sand and coarse sand, which then transitions \
        to coarse sand, gravel, and boulders deeper in the well.')

    st.header('Forecast')
    st.write('**Note, the groundwater level is given as feet from the surface.**')
    with st.expander('Information about the Models', expanded = False):
        st.markdown("""
            - Baseline: Predicts a constant value output based on the average of all observations in the training set.
            - LinReg: An ordinary least squares linear regression, fitting a linear model to the training data.
            - CNN:  neural network consisting of three consecutive one-dimensional convolutional layers, \
            followed by a sequence dense layers with rectified linear unit activation functions.
            - LSTM: A neural network consisting of a single Long Short Term Memory layer \
            followed by a sequence of dense layers with rectified linear unit activation functions.
            """)
    plot_multitraces(AEK201)

    st.header('Error')
    st.write("**Root Mean Square Error (RMSE) and Mean Average Error (MAE) for each model.**")

    fig = px.bar(AEK201_scores, x='Model', y=['RMSE', 'MAE'], barmode='group', labels={'value': 'Metric Value'})
    fig.update_layout(xaxis={'categoryorder':'total descending'},
                  xaxis_title='Model', yaxis_title='Feet from the Surface')
    st.plotly_chart(fig)
    
    st.subheader('Raw Error')
    st.table(AEK201_scores)

## AFL259 
if selected_well == "AFL259":
    ## Map (Selected blue)
    m = init_map()
    folium.Marker(location=[47.6619,-117.3655],
                 tooltip = 'AEK201',
                 icon = folium.Icon(color='red', icon="bookmark", prefix='fa')).add_to(m)
    folium.Marker(location=[47.6946,-117.1011],
                 tooltip = 'AFL259',
                 icon = folium.Icon(color='blue', icon="bookmark", prefix='fa')).add_to(m)
    folium.Marker(location=[47.698,-117.2395],
                 tooltip = 'APK309',
                 icon = folium.Icon(color='red', icon="bookmark", prefix='fa')).add_to(m)
    folium.Marker(location=[47.7007,-117.1954],
                 tooltip = 'APK310',
                 icon = folium.Icon(color='red', icon="bookmark", prefix='fa')).add_to(m)
    folium_static(m, height = 300)

    st.header('About the Well')
    st.write('AFL259 is a groundwater monitoring well in Spokane County, Washington (Latitude/Longitude: 47.6946, -117.1011) \
        at an elevation of 2,056 ft within the Columbia Plateau basin-fill aquifers. The well is 116 ft deep and the lithography varies at \
        different depths. The well generally consists of sand, gravel, and cobbles with layers of boulders throughout.')
    st.header('Forecast')
    st.write('**Note, the groundwater level is given as feet from the surface.**')
    with st.expander('Information about the Models', expanded = False):
         st.markdown("""
            - Baseline: Predicts a constant value output based on the average of all observations in the training set.
            - LinReg: An ordinary least squares linear regression, fitting a linear model to the training data.
            - CNN:  neural network consisting of three consecutive one-dimensional convolutional layers, \
            followed by a sequence dense layers with rectified linear unit activation functions.
            - LSTM: A neural network consisting of a single Long Short Term Memory layer \
            followed by a sequence of dense layers with rectified linear unit activation functions.
            """)
    plot_multitraces(AFL259)

    st.header('Error')
    st.write("**Root Mean Square Error (RMSE) and Mean Average Error (MAE) for each model.**")

    fig = px.bar(AFL259_scores, x='Model', y=['RMSE', 'MAE'], barmode='group', labels={'value': 'Metric Value'})
    fig.update_layout(xaxis={'categoryorder':'total descending'},
                  xaxis_title='Model', yaxis_title='Feet from the Surface')
    st.plotly_chart(fig)
    
    st.subheader('Raw Error')
    st.table(AFL259_scores)


## APK309
if selected_well == "APK309":
    ## Map (Selected blue)
    m = init_map()
    folium.Marker(location=[47.6619,-117.3655],
                 tooltip = 'AEK201',
                 icon = folium.Icon(color='red', icon="bookmark", prefix='fa')).add_to(m)
    folium.Marker(location=[47.6946,-117.1011],
                 tooltip = 'AFL259',
                 icon = folium.Icon(color='red', icon="bookmark", prefix='fa')).add_to(m)
    folium.Marker(location=[47.698,-117.2395],
                 tooltip = 'APK309',
                 icon = folium.Icon(color='blue', icon="bookmark", prefix='fa')).add_to(m)
    folium.Marker(location=[47.7007,-117.1954],
                 tooltip = 'APK310',
                 icon = folium.Icon(color='red', icon="bookmark", prefix='fa')).add_to(m)
    folium_static(m, height = 300)

    st.header('About the Well')
    st.write('APK309 is a groundwater monitoring well in Spokane County, Washington (Latitude/Longitude: 47.698, -117.2395) \
        at an elevation of 1,970 ft within the Columbia Plateau basin-fill aquifers. The well is 120 ft deep and \
        the lithography varies at different depths. The top 45 ft of the well consists of gravel and sand, which then \
        transitions to silt and clay down to 110 ft. The deepest 10 ft of the well is silty, gravelly sand and silty sand.')

    st.header('Forecast')
    st.write('**Note, the groundwater level is given as feet from the surface.**')
    with st.expander('Information about the Models', expanded = False):
         st.markdown("""
            - Baseline: Predicts a constant value output based on the average of all observations in the training set.
            - LinReg: An ordinary least squares linear regression, fitting a linear model to the training data.
            - CNN:  neural network consisting of three consecutive one-dimensional convolutional layers, \
            followed by a sequence dense layers with rectified linear unit activation functions.
            - LSTM: A neural network consisting of a single Long Short Term Memory layer \
            followed by a sequence of dense layers with rectified linear unit activation functions.
            """)
    plot_multitraces(APK309)

    st.header('Error')
    st.write("**Root Mean Square Error (RMSE) and Mean Average Error (MAE) for each model.**")

    fig = px.bar(APK309_scores, x='Model', y=['RMSE', 'MAE'], barmode='group', labels={'value': 'Metric Value'})
    fig.update_layout(xaxis={'categoryorder':'total descending'},
                  xaxis_title='Model', yaxis_title='Feet from the Surface')
    st.plotly_chart(fig)
    
    st.subheader('Raw Error')
    st.table(APK309_scores)

## APK310 
if selected_well == "APK310":
    ## Map (Selected blue)
    m = init_map()
    folium.Marker(location=[47.6619,-117.3655],
                 tooltip = 'AEK201',
                 icon = folium.Icon(color='red', icon="bookmark", prefix='fa')).add_to(m)
    folium.Marker(location=[47.6946,-117.1011],
                 tooltip = 'AFL259',
                 icon = folium.Icon(color='red', icon="bookmark", prefix='fa')).add_to(m)
    folium.Marker(location=[47.698,-117.2395],
                 tooltip = 'APK309',
                 icon = folium.Icon(color='red', icon="bookmark", prefix='fa')).add_to(m)
    folium.Marker(location=[47.7007,-117.1954],
                 tooltip = 'APK310',
                 icon = folium.Icon(color='blue', icon="bookmark", prefix='fa')).add_to(m)
    folium_static(m, height = 300)

    st.header("About the Well")
    st.write("APK310 is a groundwater monitoring well in Spokane County, Washington (Latitude/Longitude: 47.7007, -117.1954) \
        at an elevation of 2,062 ft within the Columbia Plateau basin-fill aquifers. The well is 125 ft deep and the lithography \
        only varies slightly at different depths. The top 5 ft of the well is sandy gravel, which then transitions to sand, gravel, \
        and cobble down to 122 ft. The deepest 3 ft of the well consists of sandy silt.")

    st.header('Forecast')
    st.write('**Note, the groundwater level is given as feet from the surface.**')
    with st.expander('Information about the Models', expanded = False):
        st.markdown("""
            - Baseline: Predicts a constant value output based on the average of all observations in the training set.
            - LinReg: An ordinary least squares linear regression, fitting a linear model to the training data.
            - CNN:  neural network consisting of three consecutive one-dimensional convolutional layers, \
            followed by a sequence dense layers with rectified linear unit activation functions.
            - LSTM: A neural network consisting of a single Long Short Term Memory layer \
            followed by a sequence of dense layers with rectified linear unit activation functions.
            """)
    plot_multitraces(APK310)

    st.header('Error')
    st.write("**Root Mean Square Error (RMSE) and Mean Average Error (MAE) for each model.**")

    fig = px.bar(APK310_scores, x='Model', y=['RMSE', 'MAE'], barmode='group', labels={'value': 'Metric Value'})
    fig.update_layout(xaxis={'categoryorder':'total descending'},
                  xaxis_title='Model', yaxis_title='Feet from the Surface')
    st.plotly_chart(fig)
    
    st.subheader('Raw Error')
    st.table(APK310_scores)

### Baseline Model







