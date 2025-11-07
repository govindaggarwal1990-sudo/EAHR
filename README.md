
# Streamlit — Employee Attrition Dashboard

A no-folder, drop-in Streamlit app (`app.py`) that lets you:
- Explore **5 rich HR charts** with filters (Job Role multi-select + satisfaction slider).
- Train/evaluate **Decision Tree, Random Forest, Gradient Boosting** with **stratified 5-fold CV**.
- View **confusion matrices**, **metrics table**, **combined ROC**, and **feature importances**.
- Upload a **new dataset** and download predictions with probabilities.

## Quick Start (Streamlit Cloud)
1. Create a new GitHub repo and upload the following files (no folders):
   - `app.py`
   - `model_utils.py`
   - `requirements.txt` *(no versions pinned)*
   - `README.md`
2. In Streamlit Cloud, link the repo and set **Main file** to `app.py`.
3. When the app loads, **upload your `EA.csv`** (must include a target `Attrition`/`attribution` column).

## Local Run (optional)
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Notes
- Filters apply to **all charts** in the Insights tab.
- Train models in the **Modeling** tab before using the **Predict** tab.
- Feature names in importance charts reflect one-hot encoded categorical levels.
