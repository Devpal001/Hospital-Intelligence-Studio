import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

# Streamlit requires page configuration before any other Streamlit command.
st.set_page_config(
    page_title="Hospital Resource Analytics",
    page_icon="🏥",
    layout="wide"
)

#custom CSS for dark theme
st.markdown("""
<style>

.main{
    background-color:#0E1117;
}

h1{
    color:#00D4FF;
    font-weight:bold;
}

h2{
    color:white;
}

h3{
    color:#90CAF9;
}

div[data-testid="stMetric"]{
    background-color:#1E1E1E;
    border:1px solid #2E86DE;
    padding:18px;
    border-radius:15px;
    box-shadow:0px 4px 15px rgba(0,0,0,0.4);
}

div[data-testid="stMetric"]:hover{
    border:1px solid #00D4FF;
}

hr{
    border:1px solid #333333;
}

</style>
""", unsafe_allow_html=True)

#loading dataset
df = pd.read_csv("hospital data analysis.csv")
#title and description
st.title("Hospital Resource Analytics Dashboard")

st.markdown(
    "Interactive dashboard for analyzing patient admissions, costs, outcomes and satisfaction."
)

# Sidebar filters

st.sidebar.title("🔍 Dashboard Filters")

gender = st.sidebar.multiselect(
    "Gender",
    df["Gender"].unique(),
    default=df["Gender"].unique()
)

condition = st.sidebar.multiselect(
    "Condition",
    df["Condition"].unique(),
    default=df["Condition"].unique()
)

outcome = st.sidebar.multiselect(
    "Outcome",
    df["Outcome"].unique(),
    default=df["Outcome"].unique()
)

age = st.sidebar.slider(
    "Age",
    int(df["Age"].min()),
    int(df["Age"].max()),
    (
        int(df["Age"].min()),
        int(df["Age"].max())
    )
)

cost = st.sidebar.slider(
    "Treatment Cost ($)",
    int(df["Cost"].min()),
    int(df["Cost"].max()),
    (
        int(df["Cost"].min()),
        int(df["Cost"].max())
    )
)

#sidebar info
st.sidebar.markdown("---")
st.sidebar.info("""
Hospital Analytics
Version 1.0
984 Patient Records
""")

#filtering logic
filtered_df = df[
    (df["Gender"].isin(gender)) &
    (df["Condition"].isin(condition)) &
    (df["Outcome"].isin(outcome)) &
    (df["Age"].between(age[0], age[1])) &
    (df["Cost"].between(cost[0], cost[1]))
]

if filtered_df.empty:
    st.info("No records match the selected filters. Adjust a filter to view the dashboard.")
    st.stop()

#loading spinner
with st.spinner("Loading Dashboard..."):

    st.success("Dashboard Loaded")

#download button

st.download_button(
    label="Download Patients Record",
    data=filtered_df.to_csv(index=False),
    file_name="filtered_hospital_data.csv",
    mime="text/csv"
)
search=st.text_input("🔍 Search Patient ID")

if search!="":
    patient_data=filtered_df[
        filtered_df["Patient_ID"].astype(str).str.contains(
            search,
            case=False
        )
    ]
else:
    patient_data=filtered_df

#Patient records
st.markdown("---")
st.subheader("📋Patient Records")

st.write(f"Showing **{len(patient_data)}** patient records matching the current filters and search.")
with st.expander("View Patient Records"):
    st.dataframe(patient_data, use_container_width=True)

#live kpi cards 
total_patients = len(filtered_df)
avg_cost = filtered_df["Cost"].mean()
avg_stay = filtered_df["Length_of_Stay"].mean()
avg_rating = filtered_df["Satisfaction"].mean()

col1, col2, col3, col4 = st.columns(4)

col1.metric("👨‍⚕️ Patients", total_patients)
col2.metric("💰 Avg Cost", f"${avg_cost:,.0f}")
col3.metric("🛏 Avg Stay (dataset units)", f"{avg_stay:.1f}")
col4.metric("⭐ Satisfaction", f"{avg_rating:.1f}/5")

# Executive Summary

st.markdown("---")
st.header("📋 Executive Summary")

st.write(f"""
This dashboard currently analyzes **{len(filtered_df)}** patient records.

### Current Filters
- **Gender:** {', '.join(gender)}
- **Condition:** {', '.join(condition)}
- **Outcome:** {', '.join(outcome)}
""")




#First chart: Patient Admissions by Condition
condition_chart = px.bar(
    filtered_df["Condition"].value_counts().reset_index(),
    x="Condition",
    y="count",
    title="Patients by Condition",
     color_discrete_sequence=["#1f77b4"]
)


#Second chart: Average Cost by Condition

cost_chart = px.bar(
    filtered_df.groupby("Condition")["Cost"].mean().reset_index(),
    x="Condition",
    y="Cost",
    title="Average Cost by Condition",
    color_discrete_sequence=["#ff7f0e"]
)


#third chart: Average Length of Stay by Condition
stay_chart = px.bar(
    filtered_df.groupby("Condition")["Length_of_Stay"].mean().reset_index(),
    x="Condition",
    y="Length_of_Stay",
    title="Average Length of Stay by Condition",
    color_discrete_sequence=["#2ca02c"]
)

#fourth chart: Average Satisfaction by Condition
satisfaction_chart = px.bar(
    filtered_df.groupby("Condition")["Satisfaction"].mean().reset_index(),
    x="Condition",
    y="Satisfaction",
    title="Average Satisfaction by Condition",
    color_discrete_sequence=["#d62728"]
)



#fifth chart: Average Cost by Outcome

cost_by_outcome_chart = px.bar(
    filtered_df.groupby("Outcome")["Cost"].mean().reset_index(),
    x="Outcome",
    y="Cost",
    title="Average Cost by Outcome",
    color="Outcome",
    color_discrete_map={"Recovered": "#2ca02c", "Stable": "#c10000"}
)


#sixth chart: Average Length of Stay by Outcome

stay_by_outcome_chart = px.bar(
    filtered_df.groupby("Outcome")["Length_of_Stay"].mean().reset_index(),
    x="Outcome",
    y="Length_of_Stay",
    title="Average Length of Stay by Outcome",
    color="Outcome",
    color_discrete_map={"Recovered": "#2ca02c", "Stable": "#c10000"}
)


#seventh chart: Average Satisfaction by Outcome

satisfaction_by_outcome_chart = px.bar(
    filtered_df.groupby("Outcome")["Satisfaction"].mean().reset_index(),
    x="Outcome",
    y="Satisfaction",
    title="Average Satisfaction by Outcome",
    color="Outcome",
    color_discrete_map={"Recovered": "#2ca02c", "Stable": "#c10000"}
)


#eighth chart: Average Cost by Gender

cost_by_gender_chart = px.bar(
    filtered_df.groupby("Gender")["Cost"].mean().reset_index(),
    x="Gender",
    y="Cost",
    title="Average Cost by Gender",
    color="Gender",
    color_discrete_map={"Male": "#1f77b4", "Female": "#ff7f0e"}
)


#ninth chart : readmission rates by condition

readmission_chart = px.histogram(
    filtered_df.groupby("Condition")["Readmission"].value_counts().reset_index(name="count"),
    x="Condition",
    y="count",
    color="Readmission",
    title="Readmission Status by Condition",
    color_discrete_map={"Yes": "#d62728", "No": "#1f77b4"}
)


#tenth chart: age distribution of patients with satisfaction ratings

satisfaction_age_chart = px.bar(
    filtered_df.groupby("Age")["Satisfaction"].mean().reset_index(),
    x="Age",
    y="Satisfaction",
    title="Age Distribution of Patients with Satisfaction Ratings",
    color_discrete_sequence=["#9467bd"]
)


# Dashboard Charts

# Row 1
col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Patient Admissions by Condition")
    st.plotly_chart(condition_chart, use_container_width=True)

with col2:
    st.subheader("💰 Average Cost by Condition")
    st.plotly_chart(cost_chart, use_container_width=True)

# Row 2
col1, col2 = st.columns(2)

with col1:
    st.subheader("🛏 Average Length of Stay by Condition")
    st.plotly_chart(stay_chart, use_container_width=True)

with col2:
    st.subheader("⭐ Average Satisfaction by Condition")
    st.plotly_chart(satisfaction_chart, use_container_width=True)

# Row 3
col1, col2 = st.columns(2)

with col1:
    st.subheader("💰 Average Cost by Outcome")
    st.plotly_chart(cost_by_outcome_chart, use_container_width=True)

with col2:
    st.subheader("🛏 Average Length of Stay by Outcome")
    st.plotly_chart(stay_by_outcome_chart, use_container_width=True)

# Row 4
col1, col2 = st.columns(2)

with col1:
    st.subheader("⭐ Average Satisfaction by Outcome")
    st.plotly_chart(satisfaction_by_outcome_chart, use_container_width=True)

with col2:
    st.subheader("💰 Average Cost by Gender")
    st.plotly_chart(cost_by_gender_chart, use_container_width=True)

# Row 5
col1, col2 = st.columns(2)

with col1:
    st.subheader("🔄 Readmission Rates by Condition")
    st.plotly_chart(readmission_chart, use_container_width=True)

with col2:
    st.subheader("👶 Age Distribution of Patients with Satisfaction Ratings")
    st.plotly_chart(satisfaction_age_chart, use_container_width=True)

#correlation heatmap

st.header("Correlation Heatmap")

fig, ax = plt.subplots(figsize=(8,5))

sns.heatmap(
    filtered_df.corr(numeric_only=True),
    annot=True,
    cmap="coolwarm",
    ax=ax
)

st.pyplot(fig)

#summary statistics

st.subheader("📊 Summary Statistics")

st.dataframe(filtered_df.describe())

#key insights

st.header("🏥 Hospital Performance Insights")

st.metric("Total Patients", len(filtered_df))
st.metric("Average Treatment Cost", f"${filtered_df['Cost'].mean():,.2f}")
st.metric("Average Hospital Stay (dataset units)", f"{filtered_df['Length_of_Stay'].mean():.1f}")
st.metric("Average Satisfaction", f"{filtered_df['Satisfaction'].mean():.1f}/5")

#financial insights
st.header("💰 Financial Insights")

highest_cost = filtered_df.groupby("Condition")["Cost"].mean().idxmax()
highest_cost_value = filtered_df.groupby("Condition")["Cost"].mean().max()

lowest_cost = filtered_df.groupby("Condition")["Cost"].mean().idxmin()
lowest_cost_value = filtered_df.groupby("Condition")["Cost"].mean().min()

st.success(
    f"💸 Highest Average Cost: **{highest_cost}** (${highest_cost_value:,.2f})"
)

st.info(
    f"💵 Lowest Average Cost: **{lowest_cost}** (${lowest_cost_value:,.2f})"
)

st.warning(
    f"💰 Total Revenue Generated: **${filtered_df['Cost'].sum():,.2f}**"
)

#disease insights

st.header("🧬 Disease Insights")

most_common = filtered_df["Condition"].value_counts().idxmax()
least_common = filtered_df["Condition"].value_counts().idxmin()

st.success(
    f"🦠 Most Common Condition: **{most_common}**"
)

st.info(
    f"🌱 Least Common Condition: **{least_common}**"
)

#patient demographics insights

st.header("👨‍⚕ Patient Demographics")

st.info(
    f"Average Patient Age: {filtered_df['Age'].mean():.1f} Years"
)

male = (filtered_df["Gender"] == "Male").sum()
female = (filtered_df["Gender"] == "Female").sum()

st.write(f"👨 Male Patients: {male}")
st.write(f"👩 Female Patients: {female}")

#readmission insights

st.header("🔄 Readmission Insights")

readmission = (
    filtered_df.groupby("Condition")["Readmission"]
    .apply(lambda x: (x == "Yes").sum())
)

highest = readmission.idxmax()

st.warning(
    f"⚠ Highest Readmission Cases: **{highest}** ({readmission.max()} Patients)"
)

rate = (filtered_df["Readmission"] == "Yes").mean() * 100

st.metric("Overall Readmission Rate", f"{rate:.1f}%")

#patient satisfaction insights

st.header("⭐ Patient Satisfaction")

best = filtered_df.groupby("Condition")["Satisfaction"].mean().idxmax()
best_score = filtered_df.groupby("Condition")["Satisfaction"].mean().max()

worst = filtered_df.groupby("Condition")["Satisfaction"].mean().idxmin()
worst_score = filtered_df.groupby("Condition")["Satisfaction"].mean().min()

st.success(
    f"😊 Highest Satisfaction: **{best}** ({best_score:.1f}/5)"
)

st.error(
    f"😟 Lowest Satisfaction: **{worst}** ({worst_score:.1f}/5)"
)

#hospital stay insights
st.header("🛏 Hospital Stay Insights")

longest = filtered_df.groupby("Condition")["Length_of_Stay"].mean().idxmax()
longest_value = filtered_df.groupby("Condition")["Length_of_Stay"].mean().max()

shortest = filtered_df.groupby("Condition")["Length_of_Stay"].mean().idxmin()
shortest_value = filtered_df.groupby("Condition")["Length_of_Stay"].mean().min()

st.success(
    f"🏥 Longest Stay: **{longest}** ({longest_value:.1f} dataset units)"
)

st.info(
    f"⚡ Shortest Stay: **{shortest}** ({shortest_value:.1f} dataset units)"
)



#footer
st.markdown("---")

st.caption(
    "Hospital Resource Analytics Dashboard | "
    "Developed using Python, Pandas, Plotly and Streamlit"
)







