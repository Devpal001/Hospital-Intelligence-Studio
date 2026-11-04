import pandas as pd
import streamlit as st


st.title("Patient Records")
st.caption("Filter and export the bundled educational dataset. It contains no documented provenance.")
df = pd.read_csv("hospital data analysis.csv")

with st.sidebar:
    st.subheader("Record filters")
    gender = st.multiselect("Gender", sorted(df["Gender"].dropna().unique()), default=sorted(df["Gender"].dropna().unique()))
    condition = st.multiselect("Condition", sorted(df["Condition"].dropna().unique()), default=sorted(df["Condition"].dropna().unique()))
    outcome = st.multiselect("Outcome", sorted(df["Outcome"].dropna().unique()), default=sorted(df["Outcome"].dropna().unique()))
    age = st.slider("Age range", int(df["Age"].min()), int(df["Age"].max()), (int(df["Age"].min()), int(df["Age"].max())))
    cost = st.slider("Treatment cost range ($)", int(df["Cost"].min()), int(df["Cost"].max()), (int(df["Cost"].min()), int(df["Cost"].max())))

search = st.text_input("Search patient ID")
mask = (
    df["Gender"].isin(gender)
    & df["Condition"].isin(condition)
    & df["Outcome"].isin(outcome)
    & df["Age"].between(*age)
    & df["Cost"].between(*cost)
)
rows = df.loc[mask]
if search:
    rows = rows[rows["Patient_ID"].astype(str).str.contains(search, case=False, na=False)]

st.write(f"**{len(rows):,}** records match your filters.")
st.dataframe(rows, hide_index=True, use_container_width=True)
st.download_button(
    "Download filtered records",
    rows.to_csv(index=False),
    file_name="filtered_hospital_records.csv",
    mime="text/csv",
    disabled=rows.empty,
)
st.info("Avoid using real patient identifiers or sensitive health information in a public portfolio version of this project.")
