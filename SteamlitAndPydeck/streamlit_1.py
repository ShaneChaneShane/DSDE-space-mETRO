import streamlit as st
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import numpy as np
import pandas as pd
import pydeck as pdk
import json

MAP_STYLES = {
    # 'Dark': 'mapbox://styles/mapbox/dark-v10',
    # 'Light': 'mapbox://styles/mapbox/light-v10',
    # 'Road': 'mapbox://styles/mapbox/streets-v11',
    # 'Satellite': 'mapbox://styles/mapbox/satellite-v9'
    'Dark': 'dark',
    'Light': 'light',
    'Road': 'road',
    'Satellite': 'satellite'
}
@st.cache_data
def load_any(file_path):
    # CSV case
    if file_path.endswith(".csv"):
        return pd.read_csv(file_path)
    
    # JSON case
    try:
        return pd.read_json(file_path)
    except ValueError:
        try:
            return pd.read_json(file_path, lines=True)
        except:
            with open(file_path, "r", encoding="utf-8") as f:
                parsed = json.loads(f.read())
            if isinstance(parsed, dict):
                return pd.DataFrame([parsed])
            return pd.DataFrame(parsed)

# Load dataset
data       = load_any("org_with_loc_v2.csv")
test_data  = load_any("OgData/test_data.json")

# train_data = load_any("train_data.json")
# raw_data   = load_any("raw_processed.json")
def time_to_color(t,alpha):
    if t < 0.5:
        ratio = t / 0.5  
        r = int(255 * ratio)      # 0  255
        g = 255                   
        b = 0
    else:
        ratio = (t - 0.5) / 0.5
        r = 255
        g = int(255 * (1 - ratio))  # 255  0
        b = 0

    return [r, g, b, alpha]

def PydeckMap(map_style,point_size,point_alpha):
    data=test_data
    min_time = data['completion_time_hours'].min()
    max_time = data['completion_time_hours'].max()
    colormap = plt.get_cmap('hsv')
    data['norm_time'] = (data['completion_time_hours'] - min_time) / (max_time - min_time)

    # # RGB + alpha
    # data['color'] = data['norm_time'].apply(
    #     lambda x: [int(c*255) for c in colormap(x)[:3]] + [20]
    # )
    data['color'] = data['norm_time'].apply(lambda t: time_to_color(t, point_alpha))


    #layer pdk
    cluster_layer = pdk.Layer(
        "ScatterplotLayer",
        data=data,  #data
        get_position=["longitude", "latitude"],  #point position
        get_radius=point_size,  #point size
        get_fill_color='color',  #point color
        pickable=True  #for tooltip
    )

    tooltip = {
        "html": "<b>Ticket ID:</b> {ticket_id} <br/>"
                "<b>Organizations:</b> {organization_array} <br/>"
                "<b>Organizations Distance:</b> {organization_distances} <br/>"
                "<b>Completion times:</b> {completion_time_hours}"
    }

    # Deck
    st.pydeck_chart(
        pdk.Deck(
            layers=[cluster_layer],
            initial_view_state=pdk.ViewState(
                latitude=data['latitude'].mean(),
                longitude=data['longitude'].mean(),
                zoom=11,
                pitch=0
            ),
            map_style=MAP_STYLES[map_style],
            tooltip=tooltip,
            height=600
        )
    )

def PydeckMap_CompleteTime_Hexagon(map_style):
    data=test_data
    # Add cluster labels to dataframe
    # normalize ค่าเวลาให้อยู่ในช่วง 0-1
    
    #layer pdk
    cluster_layer = pdk.Layer(
        "HexagonLayer",
        data=data,  #data
        get_position=["longitude", "latitude"],  #point position
        radius=100,  #point size
        get_fill_color='color',  #point color
        get_elevation="organization_distances",  # height distance
        elevation_scale=20,
        elevation_range=[0, 1000],
        pickable=True,  #for tooltip
        extruded=True ,
        pitch=45
    )

    tooltip = {
        "html": "<b>Ticket ID:</b> {ticket_id} <br/>"
                "<b>Organizations:</b> {organization_array} <br/>"
                "<b>Organizations Distance:</b> {organization_distances} <br/>"
                "<b>Completion times:</b> {completion_time_hours}"
    }

    # Deck
    st.pydeck_chart(
        pdk.Deck(
            layers=[cluster_layer],
            initial_view_state=pdk.ViewState(
                latitude=data['latitude'].mean(),
                longitude=data['longitude'].mean(),
                zoom=11,
                pitch=0
            ),
            map_style=MAP_STYLES[map_style],
            tooltip=tooltip,
            height=600
        )
    )
def PydeckMap_Heatmap(map_style):
    data=test_data
    #layer pdk
    cluster_layer = pdk.Layer(
        "HeatmapLayer",
        data=data,  #data
        get_position=["longitude", "latitude"],  #point position
        pickable=True,  #for tooltip
        get_weight="completion_time_hours"
    )

    tooltip = {
        "html": "<b>Ticket ID:</b> {ticket_id} <br/>"
                "<b>Organizations:</b> {organization_array} <br/>"
                "<b>Organizations Distance:</b> {organization_distances} <br/>"
                "<b>Completion times:</b> {completion_time_hours}"
    }

    # Deck
    st.pydeck_chart(
        pdk.Deck(
            layers=[cluster_layer],
            initial_view_state=pdk.ViewState(
                latitude=data['latitude'].mean(),
                longitude=data['longitude'].mean(),
                zoom=11,
                pitch=0
            ),
            map_style=MAP_STYLES[map_style],
            tooltip=tooltip,
            height=600
        )
    )

def main():

    st.sidebar.title("Settings")

    map_style = st.sidebar.selectbox('Select Base Map Style',options=['Dark', 'Light', 'Road', 'Satellite'],index=2)   
    point_size=st.sidebar.number_input("point size",5,100,5,step=5)
    point_alpha = st.sidebar.slider('point alpha', 1, 255, 200)


    st.title("Space Metro Project")

    PydeckMap(map_style, point_size, point_alpha)

    st.write(test_data)


if __name__ == '__main__':
    main()

# st.write("helloworld")
# st.title("im new to this")
# code='''def hello()
#     print("hello world")'''
# st.code(code ,language='python')
# isRunButton=st.button("run")
# st.write(isRunButton)
# if isRunButton :
#     st.sidebar.markdown("button pressed")
# age_input=st.number_input("input your age",0,100,10)
# st.sidebar.markdown(f"your age is {age_input}")