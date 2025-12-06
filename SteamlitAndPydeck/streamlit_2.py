import streamlit as st
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import numpy as np
import pandas as pd
import pydeck as pdk
import json
MAP_STYLES = {
    'Dark': 'dark',
    'Light': 'light',
    'Road': 'road',
    'Satellite': 'satellite'
}
# Load custom font
# st.markdown("""
# <style>
# @font-face {
#     font-family: 'Kanit';
#     src: url('font/Kanit-Medium.ttf') format('truetype');
# }

# html, body, [class*="css"] {
#     font-family: 'Kanit';
# }
# </style>
# """, unsafe_allow_html=True)

# st.title("ข้อความนี้ใช้ฟอนต์ของคุณ!")
@st.cache_data
def load_any(file_path):
    # CSV case
    if file_path.endswith(".csv"):
        df=pd.read_csv(file_path)
        df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
        df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
        return df
    
    # JSON case
    try:
        return pd.read_json(file_path)
    except:
        pass

    # Try NDJSON with chunks
    try:
        chunks = pd.read_json(file_path, lines=True)
        df = pd.concat(chunks)
        return df
    except:
        pass

    # Fallback manual load
    with open(file_path, "r", encoding="utf-8") as f:
        parsed = json.loads(f.read())
    if isinstance(parsed, dict):
        return pd.DataFrame([parsed])
    return pd.DataFrame(parsed)
@st.cache_data
def get_time_range(df):
    min_t = df['completion_time_hours'].min()
    max_t = df['completion_time_hours'].max()
    count =len(df)
    return min_t, max_t,count

# Load dataset
# OGANIZATION_DATA = load_any("org_with_loc_v2.csv")
# DATA  = load_any("OgData/raw_processed.json")
DATA  = load_any("viz9_Data.csv")
MIN_TIME, MAX_TIME,DATA_COUNT = get_time_range(DATA)
# ARC_DATA  = load_any("viz1_Data.json")

@st.cache_data
def merge_path_data():
    df = DATA.explode("organization_array").groupby("organization_array") .agg(
                 total_time=("completion_time_hours", "sum"),
                 count=("completion_time_hours", "count")
             ) .reset_index()
    df["average"] = df["total_time"] / df["count"]
    return df


def Filter_data(time_range, data_amount):
    df = DATA[
        (DATA['completion_time_hours'] >= time_range[0]) &
        (DATA['completion_time_hours'] <= time_range[1])
    ]
    return df.head(data_amount)


def Time_to_Color(t,alpha):
    if t < 0.5:
        ratio = t / 0.5  
        r = int(255 * ratio)      # 0  255 green
        g = 255                   
        b = 0
    else:
        ratio = (t - 0.5) / 0.5
        r = 255
        g = int(255 * (1 - ratio))  # 255  0 yellow
        b = 0

    return [r, g, b, alpha]

def PydeckMap(map_style,point_size,point_alpha,filtered_data,line_weight):
    
    min_t = filtered_data['completion_time_hours'].min()
    max_t = filtered_data['completion_time_hours'].max()
    filtered_data['norm_time'] = (filtered_data['completion_time_hours'] - min_t) / (max_t - min_t)
    filtered_data['color'] = filtered_data['norm_time'].apply(lambda t: Time_to_Color(t, point_alpha))


    #layer pdk
    cluster_layer = pdk.Layer(
        "ScatterplotLayer",
        data=filtered_data,  #data
        get_position=["longitude", "latitude"],  #point position
        get_radius=point_size,  #point size
        get_fill_color='color',  #point color
        pickable=True  #for tooltip
    )
    organization_layer = pdk.Layer(
        "ScatterplotLayer",
        data=filtered_data,
        get_position=["organization_longitude", "organization_latitude"],
        get_radius=100,
        get_fill_color=[255, 255, 255],   # white fill
        stroked=True,
        get_line_color=[0, 0, 0,25],  # white outline
        get_line_width=25,                 # IMPORTANT
        pickable=False
    )
    # text_layer = pdk.Layer(
    #     "TextLayer",
    #     data=OGANIZATION_DATA,
    #     get_position=["longitude", "latitude"],
    #     get_text="displayName",
    #     get_color=[0, 0, 0],       # black text
    #     get_size=8,
    #     # font_family="Kanit",
    #     get_alignment_baseline="'top'",
    # )
    arc_layer = pdk.Layer(
        "ArcLayer",
        data=filtered_data,
        get_source_position=["longitude", "latitude"],
        get_target_position=["organization_longitude","organization_latitude"],
        get_source_color='color',   
        get_target_color='color',     
        get_width=line_weight,                     
        auto_highlight=True,
        pickable=True,
    )

    # Deck No text layer for now it 
    st.pydeck_chart(
        pdk.Deck(
            layers=[cluster_layer, arc_layer,organization_layer],
            initial_view_state=pdk.ViewState(
                latitude=filtered_data['latitude'].mean(),
                longitude=filtered_data['longitude'].mean(),
                zoom=11,
                pitch=45
            ),
            map_style=MAP_STYLES[map_style],
            tooltip={
                "html": "<b>Ticket ID:</b> {ticket_id} <br/>"
                        "<b>Organizations:</b> {organization} <br/>"
                        "<b>Organizations Distance:</b> {organization_distance} <br/>"
                        "<b>Completion times:</b> {completion_time_hours}"
                },
            height=600
        )
    )
    st.write(filtered_data)
    return

def main():
#Sidebar Section Wink Wink
    st.sidebar.title("Settings")
    map_style = st.sidebar.selectbox('Select Base Map Style',options=['Dark', 'Light', 'Road', 'Satellite'],index=2)   
    data_amount = st.sidebar.slider("Number of data points",100,min(100000,DATA_COUNT),value=20000,step=1000)

    point_size=st.sidebar.number_input("point size",5,100,50,step=5)
    point_alpha = st.sidebar.slider('point alpha', 1, 255, 50)
    line_weight = st.sidebar.slider("Line Thickness",1,50,value=2,step=1)


    st.sidebar.title('Completion Time Range(hours)')
    col1, col2 = st.sidebar.columns(2)

    min_input = col1.number_input(
        "Min Hours",min_value=float(MIN_TIME),max_value=float(MAX_TIME),
        value=float(MIN_TIME),step=10.0,)
    max_input = col2.number_input(
        "Max Hours",min_value=float(MIN_TIME),max_value=float(MAX_TIME),
        value=200.0,step=10.0,)
    time_range = (min_input, max_input)
    


#Data Process
    filtered_data = Filter_data(time_range, data_amount)


#Main Section
    st.title("Space Metro Project")

    PydeckMap(map_style, point_size, point_alpha,filtered_data,line_weight)
    st.write(merge_path_data())
    # st.write(filtered_data)
    # st.write(OGANIZATION_DATA)
    return

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
# col1, col2 = st.sidebar.columns(2)
#     min_input = col1.number_input(
#         "Min Hours",value=float(MIN_TIME),min_value=float(MIN_TIME),max_value=float(MAX_TIME),step=10.0)
#     max_input = col2.number_input(
#         "Max Hours",value=200.0,min_value=float(MIN_TIME),max_value=float(MAX_TIME),step=10.0)
#     time_range = st.sidebar.slider("Select Completion Time Range(hours)",float(MIN_TIME),float(MAX_TIME),
#                                    (min_input, max_input),step=10.0)