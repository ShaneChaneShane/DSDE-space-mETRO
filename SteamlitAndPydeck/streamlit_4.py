import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import pydeck as pdk
import json
import joblib
import altair as alt
from preprocess_for_model import preprocess_for_model

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
MIN_TIME, MAX_TIME,DATA_COUNT = get_time_range(DATA)
ALL_TYPES = get_ALL_TYPES()
ALL_ORGS = get_all_organization()

@st.cache_resource
def load_model_package(path="fast_7day_model.joblib"):
    package = joblib.load(path)
    return package

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

def Merge_Path_Data(filtered_data):
    df = filtered_data.groupby("organization") .agg(
                 total_time=("completion_time_hours", "sum"),
                 count=("completion_time_hours", "count")
             ) .reset_index()
    df["average"] = df["total_time"] / df["count"]
    return df
def Plot_Completion_Time_Graph(filtered_data):
    
    merged_df = Merge_Path_Data(filtered_data)
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

def Merge_Ticket_Type_Data(filtered_data):
    df = filtered_data.copy()
    df = df.explode("type_array")
    df = df.groupby("type_array").agg(
        total_time=("completion_time_hours", "sum"),
        count=("completion_time_hours", "count")
    ).reset_index()
    df["average"] = df["total_time"] / df["count"]
    return df
def Plot_Organization_Ticket_Graph(filtered_data):
    merged_df = Merge_Ticket_Type_Data(filtered_data)
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
def Data_Pydeck_Map(map_style,point_size,point_alpha,filtered_data,line_weight):
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


def plot_half_circle_visualization(filtered_data):
    df = filtered_data.explode("type_array")
    type_stats = df.groupby("type_array").size().reset_index(name="count")

    fig = go.Figure(
        go.Pie(
            labels=type_stats["type_array"],
            values=type_stats["count"],
            hole=0.6,
            sort=False,
            direction="clockwise"
        )
    )

    fig.update_traces(rotation=180)

    fig.update_layout(
        title="Half Circle – Ticket Type Distribution",
        margin=dict(l=0, r=0, t=50, b=0),
        height=450,
        legend=dict(orientation="h", yanchor="bottom", y=-0.2)
    )

    st.plotly_chart(fig, use_container_width=True)

def main():
#Main Section
    st.title("Space Metro Project")
    st.sidebar.title("Settings")
    with st.sidebar.expander("Data Insight Settings", expanded=False):
            map_style = st.selectbox('Select Base Map Style',options=['Dark', 'Light', 'Road', 'Satellite'],index=2)   
            data_amount = st.slider("Number of data points",0,min(80000,DATA_COUNT),value=20000,step=1000)
            selected_type = st.multiselect("Types", options=ALL_TYPES, default=ALL_TYPES)
            selected_orgs = st.multiselect("Select Organizations",options=ALL_ORGS,default=ALL_ORGS)
            st.subheader('Completion Time Range(hours)')
            col1, col2 = st.columns(2)
            st.subheader('Decoration')
            point_size=st.number_input("point size",5,100,50,step=5)
            point_alpha = st.slider('point alpha', 1, 255, 50)
            line_weight = st.slider("Line Thickness",1,50,value=2,step=1)
            min_input = col1.number_input(
                "Min Hours",min_value=float(MIN_TIME),max_value=float(MAX_TIME),
                value=float(MIN_TIME),step=10.0,)
            max_input = col2.number_input(
                "Max Hours",min_value=float(MIN_TIME),max_value=float(MAX_TIME),
                value=200.0,step=10.0,)
            time_range = (min_input, max_input)

    with st.sidebar.expander("Training Result Settings", expanded=False):
        uploaded_file = st.file_uploader("Upload a cleaned CSV file", type=["csv"])


    tab_viz,tab_training= st.tabs(["Data Insight","Training Result"])
    with tab_viz:
        filtered_data = Filter_data(time_range, data_amount,selected_type,selected_orgs)
        st.header("Visualization Section")
        st.subheader("Map View")
        Data_Pydeck_Map(map_style, point_size, point_alpha,filtered_data,line_weight)
        st.subheader("Average Completion Time per Organization")
        Plot_Completion_Time_Graph(filtered_data)
        st.subheader("Average Completion Time per Type of Problem")
        Plot_Organization_Ticket_Graph(filtered_data)
    with tab_training:
        ## result from model training -overall accuracy
        #presition,recall,f1score
        #top predict feature what is the best organization for solve the citizen problem#
        #anothere thing i should plott
        from sklearn.metrics import (
            accuracy_score,
            precision_recall_fscore_support,
            classification_report,
            confusion_matrix,
            roc_auc_score,
        )

        st.title("Ticket Completion ≤7 Days Predictor")

        st.write(
            """
        This app uses your trained HistGradientBoosting model
        to predict whether a ticket will be completed within 7 days.

        """
        )

        st.set_page_config(page_title="The Ramsey Highlights", layout="wide")
        try:
            package = load_model_package()
            model = package["model"]
        except Exception as e:
            st.error(f"Could not load model package: {e}")
            st.stop()

        st.success("Model package loaded successfully.")


        if uploaded_file is None:
            st.info("Please upload a cleaned CSV file to run predictions.")
            st.stop()

        # Cache reading the CSV for speed
        @st.cache_data
        def load_data(file):
            return pd.read_csv(file)

        df_raw = load_data(uploaded_file)

        with st.spinner("Preprocessing data..."):
            try:
                X = preprocess_for_model(df_raw, package)
            except Exception as e:
                st.error(f"Error during preprocessing: {e}")
                st.stop()

        st.write("Feature matrix shape after preprocessing:", X.shape)

        # Run predictions

        with st.spinner("Running model predictions..."):
            proba = model.predict_proba(X)[:, 1]
            pred  = (proba >= 0.5).astype(int)

        # Attach predictions to original data
        results = df_raw.copy()
        results["prob_fast_7d"] = proba
        results["pred_fast_7d"] = pred  # 1 = will finish within 7 days

        cols_to_show = []
        # Show some helpful columns if they exist
        for col in ["ticket_id","timestamp", "latitude", "longitude", "organization", "type"]:
            if col in results.columns:
                cols_to_show.append(col)

        cols_to_show += ["prob_fast_7d", "pred_fast_7d"]

        def highlight_fast(row):
            if row["pred_fast_7d"] == 1:
                return ["background-color: #fff3b0"] * len(row)   # soft yellow
            else:
                return [""] * len(row)

        styled_preview = results[cols_to_show].head(20).style.apply(highlight_fast, axis=1)

        st.subheader("Predictions (first 20 rows)")
        st.dataframe(styled_preview)


        # If ground truth is available, compute evaluation metrics

        if "completion_time_hours" in results.columns:
            hours_7 = 7 * 24

            st.subheader("Evaluation on this dataset")
            # Predictions vs truths (first 20 rows)
            st.markdown("### Predictions vs Truth (first 20 rows)")

            n_show = min(20, len(results))
            first20 = results.head(n_show).copy()
            first20["true_fast_7d"] = (first20["completion_time_hours"] <= hours_7).astype(int)
            first20["correct"] = (first20["true_fast_7d"] == first20["pred_fast_7d"])

            display_cols = []
            if "ticket_id" in first20.columns:
                display_cols.append("ticket_id")
            display_cols += ["timestamp", "latitude", "longitude", "organization", "type", "completion_time_hours", "true_fast_7d", "pred_fast_7d", "prob_fast_7d", "correct"]

            # Color-code correct vs incorrect predictions
            def highlight_correct(val):
                if val is True:
                    return "background-color: #c6f5c6"  # light green
                elif val is False:
                    return "background-color: #f5c6c6"  # light red
                return ""

            styled_first20 = first20[display_cols].style.applymap(
                highlight_correct, subset=["correct"]
            )

            st.dataframe(styled_first20)
            
            # === Scatter: True vs Predicted ===
            scatter_df = first20.copy()
            scatter_df["id"] = scatter_df.index

            scatter_chart = (
                alt.Chart(scatter_df)
                .mark_circle(size=80)
                .encode(
                    x=alt.X("id:O", title="Row Index"),
                    y=alt.Y("true_fast_7d:O", title="True (0 or 1)"),
                    color=alt.Color("pred_fast_7d:N", title="Predicted"),
                    tooltip=["true_fast_7d", "pred_fast_7d"]
                )
            )

            st.markdown("### Truth vs Prediction Scatter")
            st.altair_chart(scatter_chart, use_container_width=True)

            y_true = (results["completion_time_hours"] <= hours_7).astype(int)
            y_pred = results["pred_fast_7d"]
            y_proba = results["prob_fast_7d"]

            acc = accuracy_score(y_true, y_pred)
            prec, rec, f1, _ = precision_recall_fscore_support(
                y_true, y_pred, average="binary", pos_label=1
            )
            try:
                roc_auc = roc_auc_score(y_true, y_proba)
            except Exception:
                roc_auc = float("nan")




            metrics_df = pd.DataFrame(
                {
                    "Metric": [
                        "Accuracy",
                        "Precision (fast≤7d)",
                        "Recall (fast≤7d)",
                        "F1 (fast≤7d)",
                        "ROC-AUC",
                    ],
                    "Value": [acc, prec, rec, f1, roc_auc],
                }
            ).set_index("Metric")

            st.markdown("### Overall metrics")
            st.table(metrics_df.style.format("{:.4f}"))
            # === Table for Precision / Recall / F1 for class 0 and 1 ===
            precisions, recalls, f1s, _ = precision_recall_fscore_support(
                y_true, y_pred, labels=[0, 1], zero_division=0
            )

            metric_table_df = pd.DataFrame({
                "Metric": ["Precision", "Recall", "F1"],
                "Class 0 (>7 days)": [precisions[0], recalls[0], f1s[0]],
                "Class 1 (≤7 days)": [precisions[1], recalls[1], f1s[1]]
            })

            st.markdown("### Precision / Recall / F1 by Class")
            
            numeric_cols = metric_table_df.columns[1:]   # skip "Metric"
            st.table(metric_table_df.style.format({col: "{:.4f}" for col in numeric_cols}))

            st.markdown("### Detailed classification report")

            st.text(
                classification_report(
                    y_true, y_pred, target_names=[">7 days", "≤7 days"], zero_division=0
                )
            )

            st.markdown("### Confusion matrix (rows=true, cols=pred)")

            cm = confusion_matrix(y_true, y_pred)
            cm_df = pd.DataFrame(
                cm,
                index=[">7 days", "≤7 days"],
                columns=[">7 days", "≤7 days"],
            )
            st.table(cm_df)

        else:
            st.info(
                "Column 'completion_time_hours' not found. "
                "Showing predictions only (no evaluation metrics)."
            )

        st.subheader("Download predictions")

        csv_bytes = results[cols_to_show].to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download results as CSV",
            data=csv_bytes,
            file_name="ticket_predictions.csv",
            mime="text/csv",)


if __name__ == '__main__':
    main()

