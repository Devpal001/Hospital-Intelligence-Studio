import pandas as pd
import plotly.express as px
import streamlit as st


st.title("Hospital Insights")
st.caption("Descriptive summaries from the bundled dataset. These are associations, not causal findings.")
df = pd.read_csv("hospital data analysis.csv")

condition = st.multiselect("Conditions", sorted(df["Condition"].unique()), default=sorted(df["Condition"].unique()))
view = df[df["Condition"].isin(condition)]
if view.empty:
    st.info("Select at least one condition to view summaries.")
    st.stop()

overall_readmission = (view["Readmission"] == "Yes").mean()
recovery_rate = (view["Outcome"] == "Recovered").mean()
a, b, c, d = st.columns(4)
a.metric("Records", f"{len(view):,}")
b.metric("Mean cost", f"${view['Cost'].mean():,.0f}")
c.metric("Readmission rate", f"{overall_readmission:.1%}")
d.metric("Recovered", f"{recovery_rate:.1%}")

summary = view.groupby("Condition", as_index=False).agg(
    patients=("Patient_ID", "size"),
    mean_cost=("Cost", "mean"),
    mean_stay=("Length_of_Stay", "mean"),
    readmission_rate=("Readmission", lambda values: (values == "Yes").mean()),
    mean_satisfaction=("Satisfaction", "mean"),
)
left, right = st.columns(2)
with left:
    st.plotly_chart(px.bar(summary, x="Condition", y="readmission_rate", title="Readmission rate by condition", labels={"readmission_rate": "Rate"}), use_container_width=True)
with right:
    st.plotly_chart(px.scatter(summary, x="mean_cost", y="mean_satisfaction", size="patients", color="Condition", title="Average cost and satisfaction", labels={"mean_cost": "Mean cost ($)", "mean_satisfaction": "Mean satisfaction"}), use_container_width=True)
st.subheader("Condition summary")
st.dataframe(summary.round(3), hide_index=True, use_container_width=True)
st.warning("The data source and length-of-stay unit are undocumented. Treat these as practice summaries, not hospital benchmarks.")
