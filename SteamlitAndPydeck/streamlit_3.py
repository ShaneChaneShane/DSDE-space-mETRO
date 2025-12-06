import streamlit as st
import plotly.express as px
import pandas as pd
import pydeck as pdk
import json

st.markdown("""
<style>
@font-face {
    font-family: 'Kanit';
    src: url('./fonts/Kanit-Medium.ttf') format('truetype');
    font-weight: normal;
    font-style: normal;
}

[data-testid="stText"] {
    font-family: 'Kanit', sans-serif;
}

body, div, span, h1, h2, h3, h4, h5, h6, p, button, label {
    font-family: 'Kanit', sans-serif ;
}

.css-1emrehy, .css-1avcm0n {
    font-family: 'Kanit', sans-serif;
}
.thai-text {
    font-family: 'Kanit', sans-serif;
    color: red;
    font-size: 14px;
    position: absolute;
}
</style>
""", unsafe_allow_html=True)

MAP_STYLES = {
    'Dark': 'dark',
    'Light': 'light',
    'Road': 'road',
    'Satellite': 'satellite'
}
@st.cache_data
def load_any(file_path):
    # CSV case
    if file_path.endswith(".csv"):
        df=pd.read_csv(file_path)
        df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
        df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
        # แปลง type_array จาก string เป็น list
        df['type_array'] = df['type_array'].apply(lambda x: [i.strip() for i in str(x).split(",")])

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
@st.cache_data
def get_ALL_TYPES():
    return sorted(set([t for sublist in DATA['type_array'] for t in sublist]))
@st.cache_data
def get_all_organization():
    return DATA['organization'].unique()
DATA  = load_any("viz9_Data.csv")
# DATA['latitude'] = pd.to_numeric(DATA['latitude'], errors='coerce')
# DATA['longitude'] = pd.to_numeric(DATA['longitude'], errors='coerce')
# DATA['organization_latitude'] = pd.to_numeric(DATA['organization_latitude'], errors='coerce')
# DATA['organization_longitude'] = pd.to_numeric(DATA['organization_longitude'], errors='coerce')
# DATA['completion_time_hours'] = pd.to_numeric(DATA['completion_time_hours'], errors='coerce')
# DATA = DATA.dropna(subset=['longitude', 'latitude', 'organization_longitude', 'organization_latitude'])


MIN_TIME, MAX_TIME,DATA_COUNT = get_time_range(DATA)
ALL_TYPES = get_ALL_TYPES()
ALL_ORGS = get_all_organization()

def Filter_data(time_range, data_amount,selected_type,selected_orgs):
    df = DATA[
        (DATA['completion_time_hours'] >= time_range[0]) &
        (DATA['completion_time_hours'] <= time_range[1])
    ]
    if selected_type and len(selected_type) > 0:
        df = df[df['type_array'].apply(lambda types: any(t in types for t in selected_type))]
    
    if selected_orgs and len(selected_orgs) > 0:
        df = df[df['organization'].isin(selected_orgs)]
    
    return df.head(data_amount)

def merge_path_data(filtered_data):
    df = filtered_data.groupby("organization") .agg(
                 total_time=("completion_time_hours", "sum"),
                 count=("completion_time_hours", "count")
             ) .reset_index()
    df["average"] = df["total_time"] / df["count"]
    return df
def plot_completion_time_graph(filtered_data):
    
    merged_df = merge_path_data(filtered_data)
    short_labels = [
        (name[:10] + "…") if len(name) > 10 else name
        for name in merged_df["organization"]]
    fig = px.bar(
        merged_df,
        x="organization",
        y="count",
        color="average",
        hover_data=["total_time", "count"],
        labels={"average": "Avg(hours)", "organization": "Organization"},
        # title="Average Completion Time per Organization",
        height=600,
    )
    fig.update_xaxes(
        ticktext=short_labels,
        tickvals=merged_df["organization"],
        # tickangle=45
    )

    fig.update_layout(
        xaxis=dict(
            categoryorder="array",
            categoryarray=merged_df.sort_values("count", ascending=False)["organization"],
            tickfont=dict(size=10)
        ),
        bargap=0.1,
        xaxis_tickangle=45 
    )

    st.plotly_chart(fig, use_container_width=True)
    st.write(merged_df)
    return 

def merge_ticket_type_data(filtered_data):
    df = filtered_data.copy()

    df = df.explode("type_array")
    df = df.groupby("type_array").agg(
        total_time=("completion_time_hours", "sum"),
        count=("completion_time_hours", "count")
    ).reset_index()
    df["average"] = df["total_time"] / df["count"]
    return df

def plot_organization_ticket_graph(filtered_data):
    merged_df = merge_ticket_type_data(filtered_data)
    short_labels = [
        (name[:10] + "…") if len(name) > 10 else name
        for name in merged_df["type_array"]]
    fig = px.bar(
        merged_df,
        x="type_array",
        y="count",
        color="average",
        color_continuous_scale="Greens",
        hover_data=["total_time", "count"],
        labels={"average": "Avg(hours)", "type_array": "Type_array"},
        height=600,
    )
    fig.update_xaxes(
        ticktext=short_labels,
        tickvals=merged_df["type_array"],
        # tickangle=45
    )

    fig.update_layout(
        xaxis=dict(
            categoryorder="array",
            categoryarray=merged_df.sort_values("count", ascending=False)["type_array"],
            tickfont=dict(size=10)
        ),
        bargap=0.1,
        xaxis_tickangle=45 
    )




    st.plotly_chart(fig, use_container_width=True)
    st.write(merged_df)
    return 


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
    #     data=filtered_data,
    #     get_position=["organization_longitude", "organization_latitude"],
    #     get_text="organization",
    #     get_color=[0, 0, 0,5],       # black text
    #     get_size=8,
    #     get_font_family="Monospace",
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
    data_amount = st.sidebar.slider("Number of data points",0,min(80000,DATA_COUNT),value=20000,step=1000)
    selected_type = st.sidebar.multiselect("Types", options=ALL_TYPES, default=ALL_TYPES)
    selected_orgs = st.sidebar.multiselect("Select Organizations",options=ALL_ORGS,default=ALL_ORGS)
    st.sidebar.subheader('Completion Time Range(hours)')
    col1, col2 = st.sidebar.columns(2)
    st.sidebar.subheader('Decoration')
    point_size=st.sidebar.number_input("point size",5,100,50,step=5)
    point_alpha = st.sidebar.slider('point alpha', 1, 255, 50)
    line_weight = st.sidebar.slider("Line Thickness",1,50,value=2,step=1)


    min_input = col1.number_input(
        "Min Hours",min_value=float(MIN_TIME),max_value=float(MAX_TIME),
        value=float(MIN_TIME),step=10.0,)
    max_input = col2.number_input(
        "Max Hours",min_value=float(MIN_TIME),max_value=float(MAX_TIME),
        value=200.0,step=10.0,)
    time_range = (min_input, max_input)
    

#Data Process
    filtered_data = Filter_data(time_range, data_amount,selected_type,selected_orgs)


#Main Section
    st.title("Space Metro Project")
    tab_viz, tab_analysis = st.tabs(["Visualization", "Model Analysis"])
    with tab_viz:
        st.header("Visualization Section")
        st.subheader("Map View")
        PydeckMap(map_style, point_size, point_alpha,filtered_data,line_weight)
        st.subheader("Average Completion Time per Organization")
        plot_completion_time_graph(filtered_data)
        st.subheader("Average Completion Time per Type of Problem")
        plot_organization_ticket_graph(filtered_data)

# Model section
    with tab_analysis:

        st.header("Model Section")
        st.subheader("Merged Data Summary")
        return

if __name__ == '__main__':
    main()

