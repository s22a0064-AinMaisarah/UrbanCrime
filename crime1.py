import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# --- DATA URL ---
CSV_URL = 'https://raw.githubusercontent.com/s22a0064-AinMaisarah/UrbanCrime/refs/heads/main/df_uber_cleaned.csv'
ENCODING_TYPE = 'cp1252'

# --- APP TITLE ---
st.title("🔍 Clustering Cities Based on Crime Patterns")
st.markdown("Using machine learning (K-Means & PCA) to uncover crime behavior patterns in cities.")
st.markdown("---")

# --- OBJECTIVE STATEMENT ---
st.subheader("🎯 Objective")
st.write("""
The objective of using K-Means clustering is to group cities into three distinct clusters based on their crime profiles including violent, property, white-collar, and social crimes,
so that cities with similar crime patterns are categorized together.
This allows for clear comparison between areas with different crime characteristics and supports targeted crime-prevention strategies.
""")

# --- LOAD DATA ---
@st.cache_data
def load_data(url, encoding):
    try:
        df = pd.read_csv(url, encoding=encoding)
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

df = load_data(CSV_URL, ENCODING_TYPE)

if not df.empty:


    
    # --- SUMMARY METRICS BOX ---
    st.subheader("📊 Crime Dataset Overview")

    col1, col2, col3, col4 = st.columns(4)

    num_cities = df['city_cat'].nunique() if 'city_cat' in df.columns else len(df)
    avg_violent = df['violent_crime'].mean().round(2)
    avg_property = df['property_crime'].mean().round(2)
    avg_whitecollar = df['whitecollar_crime'].mean().round(2)
    avg_social = df['social_crime'].mean().round(2)

    col1.metric("🏙️ Cities Analyzed", value=num_cities)
    col2.metric("⚠️ Avg Violent Crime", value=avg_violent)
    col3.metric("🏚️ Avg Property Crime", value=avg_property)
    col4.metric("💼 Avg White-Collar Crime", value=avg_whitecollar)

    col5, col6 = st.columns(2)
    col5.metric("👥 Avg Social Crime", value=avg_social)
    col6.metric("📂 Clusters Formed", value="3 (K-Means)")

    st.markdown("---")
    
    # --- SELECT FEATURES ---
    features = ['violent_crime', 'property_crime', 'whitecollar_crime', 'social_crime']
    X = df[features]

    # --- STANDARDIZE FEATURES ---
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # --- ELBOW METHOD ---
    wcss = []
    for k in range(2, 10):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        wcss.append(kmeans.inertia_)

    # --- ELBOW PLOT ---
    st.subheader("📈 Elbow Method: Optimal Number of Clusters")
    fig, ax = plt.subplots()
    ax.plot(range(2, 10), wcss, marker='o')
    ax.set_title("Elbow Method for Optimal k")
    ax.set_xlabel("Number of Clusters (k)")
    ax.set_ylabel("WCSS")
    st.pyplot(fig)

    # --- APPLY KMEANS (k=3) ---
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df['crime_cluster'] = kmeans.fit_predict(X_scaled)

    # --- PCA ---
    pca = PCA(n_components=2)
    pca_data = pca.fit_transform(X_scaled)
    df['PC1'] = pca_data[:, 0]
    df['PC2'] = pca_data[:, 1]

    # --- PCA VISUALIZATION ---
    st.subheader("📊 Crime Pattern Cluster Visualization (PCA)")
    fig2, ax2 = plt.subplots()
    sns.scatterplot(x='PC1', y='PC2', hue='crime_cluster', data=df, palette='viridis', s=120, ax=ax2)
    ax2.set_title("Crime Pattern Clusters (PCA)")
    st.pyplot(fig2)

    # --- CLUSTER PROFILE ---
    cluster_profile = df.groupby('crime_cluster')[features].mean()

    st.subheader("🏙️ Crime Type Average per Cluster")
    fig3, ax3 = plt.subplots()
    cluster_profile.T.plot(kind='bar', ax=ax3)
    ax3.set_title("Crime Category Levels by Cluster")
    ax3.set_ylabel("Scaled Crime Level")
    ax3.set_xticklabels(ax3.get_xticklabels(), rotation=0)
    st.pyplot(fig3)

    # --- INTERPRETATION ---
    st.subheader("🧠 Interpretation & Discussion")
    st.write("""
- The Elbow chart suggests **3 clusters** as the optimal grouping.
- PCA scatter plot shows **clear separation** among clusters, indicating meaningful patterns.
- Cluster profiles reveal different crime dominance patterns:
  - Some cities experience **higher violent & property crimes**
  - Other areas have **elevated white-collar or social crimes**
- These insights help authorities:
  ✅ identify risk zones  
  ✅ plan targeted policing  
  ✅ allocate resources efficiently  
""")

else:
    st.error("Failed to load dataset.")
