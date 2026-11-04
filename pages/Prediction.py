import pandas as pd
import streamlit as st

from components.prediction import FEATURES, feature_influence, predict_all, train_target_models


st.title("Length-of-Stay Forecast")
st.caption("Your seven-factor multivariate linear regression model, connected to an interactive scenario form.")
st.warning(
    "The training CSV referenced by your original script is not present in this workspace. "
    "This uses the saved coefficients and has not been independently validated."
)

weights = [0.09271369, 1.07433568, 0.68028327, 0.55618327, 0.33216646, 1.00949448, 0.24837546]
bias = 0.6651


@st.cache_resource(show_spinner="Comparing auxiliary models with cross-validation…")
def get_aux_models(data):
    return train_target_models(data)

with st.form("length_of_stay_model"):
    a, b = st.columns(2)
    with a:
        age = st.number_input("Age", min_value=0.0, max_value=120.0, value=65.0)
        comorbidities = st.number_input("Comorbidities count", min_value=0.0, max_value=50.0, value=2.0)
        surgery = st.selectbox("Surgery flag", [0, 1], format_func=lambda value: "Yes (1)" if value else "No (0)")
        icu = st.selectbox("ICU flag", [0, 1], format_func=lambda value: "Yes (1)" if value else "No (0)")
    with b:
        infection = st.selectbox("Hospital-acquired infection", [0, 1], format_func=lambda value: "Yes (1)" if value else "No (0)")
        delay = st.number_input("Discharge delay (days)", min_value=0.0, max_value=365.0, value=0.0)
        support = st.number_input("Social support value", min_value=0.0, max_value=100.0, value=1.0, help="Enter the encoding used when your model was trained.")
    estimate = st.form_submit_button("Estimate length of stay", type="primary", use_container_width=True)

if estimate:
    values = [age, comorbidities, surgery, icu, infection, delay, support]
    predicted_days = sum(value * weight for value, weight in zip(values, weights)) + bias
    st.metric("Estimated length of stay", f"{predicted_days:.1f} days")
    st.caption("Seven-factor linear regression using the coefficients saved in hospitalbedoccupancypredictor.py.")

st.info(
    "Confirm the input units and social-support encoding against the training data before interpreting the estimate. "
    "The result is for learning only, not a clinical or bed-allocation decision."
)

with st.expander("Auxiliary sample-data outcome models"):
    st.caption("Separate exploratory predictions trained on the bundled 984-row file. Its source is undocumented and patterns are unusually regular.")
    if st.checkbox("Open auxiliary models"):
        df = pd.read_csv("hospital data analysis.csv")
        required = set(FEATURES + ["Readmission", "Outcome", "Satisfaction", "Cost", "Length_of_Stay"])
        missing = required - set(df.columns)
        if missing:
            st.error(f"The dataset is missing required columns: {', '.join(sorted(missing))}")
        else:
            model_results = get_aux_models(df)
            st.subheader("Auxiliary patient scenario")
            with st.form("aux_patient_prediction"):
                left, right = st.columns(2)
                with left:
                    aux_age = st.slider("Age", int(df["Age"].min()), int(df["Age"].max()), int(df["Age"].median()), key="aux_age")
                    gender = st.selectbox("Gender", sorted(df["Gender"].dropna().unique()), key="aux_gender")
                with right:
                    condition = st.selectbox("Condition", sorted(df["Condition"].dropna().unique()), key="aux_condition")
                    procedures = sorted(df.loc[df["Condition"] == condition, "Procedure"].dropna().unique())
                    procedure = st.selectbox("Procedure", procedures, key="aux_procedure")
                submitted = st.form_submit_button("Generate auxiliary estimates")
            if submitted:
                predictions = predict_all(model_results, {"Age": aux_age, "Gender": gender, "Condition": condition, "Procedure": procedure})
                cards = st.columns(5)
                cards[0].metric("Readmission", str(predictions["Readmission"]["prediction"]))
                cards[1].metric("Outcome", str(predictions["Outcome"]["prediction"]))
                cards[2].metric("Expected cost", f"{predictions['Cost']['prediction']:,.0f} dataset units")
                cards[3].metric("Expected stay", f"{predictions['Length_of_Stay']['prediction']:.1f} dataset units")
                cards[4].metric("Satisfaction", f"{predictions['Satisfaction']['prediction']:.0f}/5")
            st.subheader("Cross-validation comparison")
            for target, result in model_results.items():
                with st.expander(f"{target}: selected {result.model_name}"):
                    st.dataframe(pd.DataFrame(result.comparison).set_index("model"), use_container_width=True)
                    st.dataframe(feature_influence(result), hide_index=True, use_container_width=True)
