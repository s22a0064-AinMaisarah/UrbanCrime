import streamlit as st

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Urban Crime Data Intelligence Platform",
    page_icon="🚨",
    layout="wide",
)

# --- HEADER ---
st.title("🚨 Urban Crime Data Intelligence Platform")
st.markdown("""
Welcome to the **Urban Crime Analytics System**, where data-driven insights 
help uncover hidden patterns behind city crime trends.
""")
st.markdown("---")

# --- INTRO SECTION ---
st.subheader("🔍 Overview")
st.write("""
This platform integrates **data visualization** and **machine learning (K-Means & PCA)** 
to analyze and cluster urban crime behavior.  

Through the dashboard, users can:
- Examine **crime distribution** across cities.
- Identify **clusters** of similar crime patterns.
- Explore **demographic influences** such as **gender**, **age**, and **education** on crime.
""")

st.markdown("---")

# --- DASHBOARD PREVIEW ---
st.subheader("📊 Dashboard at a Glance")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Cities Analyzed", "Multiple", help="Total number of cities included in the dataset.")
with col2:
    st.metric("Crime Categories", "4 Types", help="Violent, Property, White-collar, and Social Crimes.")
with col3:
    st.metric("ML Model", "K-Means Clustering", help="Used to group cities by crime patterns.")

st.markdown("---")

# --- NAVIGATION SECTION ---
st.subheader("📂 Navigation Guide")
st.info("""
Use the **left sidebar** to explore:
1️⃣ **Crime Cluster Analysis** – Explore city-level clustering (Machine Learning view).  
2️⃣ **Demographic Insights** – Analyze crime trends by gender, age, and education.  
""")

st.markdown("---")
st.success("✅ Tip: Click the sidebar menu to begin your analysis journey!")
