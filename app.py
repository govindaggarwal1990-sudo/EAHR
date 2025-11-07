
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import plotly.express as px

from model_utils import (
    prepare_data, train_all_models, evaluate_models,
    run_inference_on_new_data, build_preprocessor
)

st.set_page_config(page_title="Employee Attrition — HR Analytics", layout="wide")

# --------------- Sidebar: data source ----------------
st.sidebar.header("Data")
uploaded = st.sidebar.file_uploader("Upload your EA.csv (with Attrition/attribution column)", type=["csv"])
use_cache = st.sidebar.checkbox("Cache computations", value=True)

@st.cache_data(show_spinner=False)
def load_df(file):
    return pd.read_csv(file)

if uploaded is not None:
    df = load_df(uploaded)
else:
    st.info("Upload your dataset to start (CSV expected). The app will look for a target column named 'Attrition' (case-insensitive).")
    st.stop()

# Try to find target
possible_targets = ["Attrition", "attrition", "Attribution", "attribution"]
target_col = None
for c in df.columns:
    if c in possible_targets or c.lower() in {"attrition", "attribution"}:
        target_col = c
        break
if target_col is None:
    st.error("Target column 'Attrition' (or 'attribution') not found.")
    st.stop()

# Identify useful columns
cat_cols = df.drop(columns=[target_col]).select_dtypes(include=["object", "category"]).columns.tolist()
num_cols = df.drop(columns=[target_col]).select_dtypes(include=[np.number, "bool"]).columns.tolist()

# ----------------- Filters -----------------
st.sidebar.header("Filters (apply to all charts)")
job_col_guess = "JobRole" if "JobRole" in df.columns else (cat_cols[0] if len(cat_cols) else None)
if job_col_guess is None:
    st.warning("No categorical columns found to filter by job role. Filters disabled.")
    selected_jobs = []
else:
    options = sorted(df[job_col_guess].dropna().unique().tolist())
    selected_jobs = st.sidebar.multiselect(f"Filter by {job_col_guess}", options, default=options)

# Satisfaction slider (choose which satisfaction column to use)
sat_candidates = [c for c in df.columns if "satisfaction" in c.lower() or c in ["JobSatisfaction","EnvironmentSatisfaction","RelationshipSatisfaction","WorkLifeBalance"]]
sat_col = None
if len(sat_candidates) > 0:
    sat_col = st.sidebar.selectbox("Choose satisfaction-related column", options=sat_candidates, index=0)
    vmin, vmax = int(df[sat_col].min()), int(df[sat_col].max())
    threshold = st.sidebar.slider(f"Min {sat_col}", vmin, vmax, value=vmin, step=1)
else:
    threshold = None

# Apply filters
filtered = df.copy()
if job_col_guess is not None and len(selected_jobs) > 0:
    filtered = filtered[filtered[job_col_guess].isin(selected_jobs)]
if sat_col is not None and threshold is not None:
    filtered = filtered[filtered[sat_col] >= threshold]

st.title("Employee Attrition — HR Analytics Dashboard")
st.caption("Interactive insights, modeling, and prediction for HR decision-making")

tab1, tab2, tab3 = st.tabs(["📊 Insights", "🤖 Modeling (DT / RF / GB)", "📤 Predict on New Data"])

# --------------------- INSIGHTS TAB ---------------------
with tab1:
    st.subheader("Overview & Filters")
    colA, colB, colC = st.columns(3)
    with colA:
        total = len(filtered)
        st.metric("Employees (after filters)", total)
    with colB:
        rate = None
        if filtered[target_col].nunique() == 2:
            pos_rate = (filtered[target_col] == filtered[target_col].unique()[-1]).mean()
            rate = f"{100*pos_rate:.1f}%"
        st.metric("Attrition Rate", rate if rate is not None else "—")
    with colC:
        st.metric("Columns", len(filtered.columns))

    # CHART 1: Attrition rate by JobRole
    st.markdown("#### 1) Attrition rate by Job Role (normalized stack)")
    if job_col_guess is not None:
        df1 = filtered.groupby([job_col_guess, target_col]).size().reset_index(name="count")
        df1["total"] = df1.groupby(job_col_guess)["count"].transform("sum")
        df1["rate"] = df1["count"] / df1["total"]
        chart1 = alt.Chart(df1).mark_bar().encode(
            x=alt.X(f"{job_col_guess}:N", title=job_col_guess, sort='-y'),
            y=alt.Y("rate:Q", stack="normalize", axis=alt.Axis(format='%')),
            color=alt.Color(f"{target_col}:N"),
            tooltip=[job_col_guess, target_col, alt.Tooltip("rate:Q", format=".1%"), "count:Q"]
        ).properties(height=380)
        st.altair_chart(chart1, use_container_width=True)
    else:
        st.info("JobRole-like column not available for this chart.")

    # CHART 2: Income distribution vs Attrition
    st.markdown("#### 2) Monthly Income distribution by Attrition")
    num_income = "MonthlyIncome" if "MonthlyIncome" in filtered.columns else (num_cols[0] if len(num_cols) else None)
    if num_income is not None:
        fig2 = px.box(filtered, x=target_col, y=num_income, points="all", title="Income distribution by Attrition")
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("No numeric column for income found.")

    # CHART 3: Heatmap — Attrition rate by Overtime × JobLevel
    st.markdown("#### 3) Heatmap: Attrition rate by Overtime × JobLevel")
    if "OverTime" in filtered.columns and "JobLevel" in filtered.columns and filtered[target_col].nunique()==2:
        rate = (filtered.assign(pos=(filtered[target_col] == filtered[target_col].unique()[-1]).astype(int))
                      .groupby(["OverTime","JobLevel"])["pos"].mean().reset_index(name="rate"))
        heatfig = px.density_heatmap(rate, x="OverTime", y="JobLevel", z="rate", color_continuous_scale="Blues")
        heatfig.update_layout(coloraxis_colorbar_title="Attrition rate")
        st.plotly_chart(heatfig, use_container_width=True)
    else:
        st.info("OverTime and/or JobLevel not available.")

    # CHART 4: Distance from Home vs Attrition (violin)
    st.markdown("#### 4) Distance From Home vs Attrition (violin)")
    if "DistanceFromHome" in filtered.columns:
        fig4 = px.violin(filtered, x=target_col, y="DistanceFromHome", box=True, points="all")
        st.plotly_chart(fig4, use_container_width=True)
    else:
        st.info("Column 'DistanceFromHome' not available.")

    # CHART 5: Attrition by tenure bins
    st.markdown("#### 5) Attrition rate by tenure (YearsAtCompany bins)")
    if "YearsAtCompany" in filtered.columns and filtered[target_col].nunique()==2:
        temp = filtered.copy()
        temp["TenureBin"] = pd.cut(temp["YearsAtCompany"], bins=[-0.1,1,3,5,10,40], labels=["≤1","2–3","4–5","6–10","10+"])
        rate_df = temp.assign(pos=(temp[target_col]==temp[target_col].unique()[-1]).astype(int))                      .groupby("TenureBin")["pos"].mean().reset_index(name="AttritionRate")
        line = alt.Chart(rate_df).mark_line(point=True).encode(
            x=alt.X("TenureBin:N", sort=["≤1","2–3","4–5","6–10","10+"]),
            y=alt.Y("AttritionRate:Q", axis=alt.Axis(format='%')),
            tooltip=["TenureBin", alt.Tooltip("AttritionRate:Q", format=".1%")]
        ).properties(height=360)
        st.altair_chart(line, use_container_width=True)
    else:
        st.info("YearsAtCompany not available or target not binary.")

# --------------------- MODELING TAB ---------------------
with tab2:
    st.subheader("Train & Evaluate — Decision Tree, Random Forest, Gradient Boosting")
    col1, col2 = st.columns([1,3])
    with col1:
        test_size = st.slider("Test size", 0.1, 0.4, value=0.2, step=0.05)
        random_state = st.number_input("Random state", value=42, step=1)
        if st.button("Run 5-fold CV + Train Models"):
            with st.spinner("Training models with 5-fold Stratified CV..."):
                X, y, classes, cat_cols, num_cols = prepare_data(df, target_col)
                results, plots, fitted = evaluate_models(
                    X, y, classes,
                    test_size=test_size, random_state=random_state, cv_splits=5
                )
                st.session_state["results"] = results
                st.session_state["plots"] = plots
                st.session_state["fitted"] = fitted
    if "results" in st.session_state:
        st.success("Models trained. See the outputs below.")
        st.dataframe(st.session_state["results"])
        for label, fig in st.session_state["plots"].items():
            st.markdown(f"**{label}**")
            st.pyplot(fig)

# --------------------- PREDICT TAB ---------------------
with tab3:
    st.subheader("Predict on New Data")
    st.caption("Upload a new CSV. The app will apply the preprocessor & best model fitted in the Modeling tab.")
    newfile = st.file_uploader("Upload new CSV for prediction", key="predict_uploader", type=["csv"])
    if newfile is not None:
        if "fitted" not in st.session_state:
            st.warning("Please train models in the Modeling tab first.")
        else:
            newdf = pd.read_csv(newfile)
            best_name = st.selectbox("Choose a trained model", options=list(st.session_state["fitted"].keys()))
            yhat, proba, out = run_inference_on_new_data(
                newdf, st.session_state["fitted"][best_name],
                label_positive=None, proba_colname="Attrition_Prob"
            )
            st.dataframe(out.head(50))
            csv = out.to_csv(index=False).encode("utf-8")
            st.download_button("Download predictions CSV", data=csv, file_name="predictions_with_attrition.csv", mime="text/csv")
