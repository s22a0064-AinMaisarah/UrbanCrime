import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Urban Crime Demographic Insights",
    page_icon="👥",
    layout="wide",
)

# --- HEADER ---
st.title("Urban Crime Demographic Insights Dashboard")
st.markdown("""
Exploring how **gender**, **age**, and **education levels** influence different types of urban crimes.
""")
st.markdown("---")

# --- LOAD DATA ---
@st.cache_data
def load_data(url, encoding):
    try:
        df = pd.read_csv(url, encoding=encoding)
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

# --- DATA URL ---
CSV_URL = 'https://raw.githubusercontent.com/s22a0064-AinMaisarah/UrbanCrime/refs/heads/main/df_uber_cleaned.csv'
ENCODING_TYPE = 'cp1252'

df_uber_cleaned = load_data(CSV_URL, ENCODING_TYPE)

# --- MAIN APP ---
if not df_uber_cleaned.empty:

    # === OBJECTIVE ===
    st.subheader("Objective")
    st.write("""
    The objective of this analysis is to understand how **demographic factors** such as gender ratio, age group, 
    and education attainment influence the distribution of different crime types.
    
    By analyzing these relationships, we can:
    - Identify demographic patterns correlated with specific crime categories.  
    - Support data-driven policymaking for targeted awareness and prevention programs.  
    - Uncover trends that connect **social structure** and **urban safety**.
    """)
    st.markdown("---")

    # === GENDER VS CRIME ===
    st.subheader("1️⃣ Gender Category and Crime Type")

    male_threshold = df_uber_cleaned['male'].mean()
    df_uber_cleaned['gender_category'] = df_uber_cleaned['male'].apply(
        lambda x: 'High-Male' if x > male_threshold else 'Balanced-Gender'
    )

    crime_scores_melted_gender = df_uber_cleaned.melt(
        id_vars='gender_category',
        value_vars=['violent_crime', 'property_crime', 'whitecollar_crime', 'social_crime'],
        var_name='Crime Type',
        value_name='Average Crime Score'
    )

    fig1, ax1 = plt.subplots(figsize=(10, 6))
    sns.barplot(
        x='gender_category', y='Average Crime Score', hue='Crime Type',
        data=crime_scores_melted_gender, ax=ax1, palette='coolwarm'
    )
    ax1.set_title('Average Crime Scores by Gender Category and Crime Type', fontsize=13)
    ax1.set_xlabel('Gender Category')
    ax1.set_ylabel('Average Crime Score')
    st.pyplot(fig1)

    st.info("""
    **Insight:**  
    - Cities with a **higher male population** often exhibit increased levels of **violent** and **property crimes**.  
    - In contrast, **balanced-gender** cities tend to show more uniform crime distributions across categories.
    """)

    st.markdown("---")

    # === AGE GROUP VS CRIME (RADAR CHART) ===
    st.subheader("2️⃣ Age Group and Crime Type")

    crime_cols = ['violent_crime', 'property_crime', 'whitecollar_crime', 'social_crime']
    age_group_crime_means = df_uber_cleaned.groupby('age')[crime_cols].mean().reset_index()

    num_vars = len(crime_cols)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    fig2, ax2 = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    for index, row in age_group_crime_means.iterrows():
        values = row[crime_cols].tolist()
        values += values[:1]
        ax2.plot(angles, values, label=f'Age Group {row["age"]}')
        ax2.fill(angles, values, alpha=0.1)

    ax2.set_thetagrids(np.degrees(angles[:-1]), crime_cols)
    ax2.set_title('Average Crime Scores by Age Group and Crime Type', size=14, y=1.1)
    ax2.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    ax2.grid(True)
    st.pyplot(fig2)

    st.info("""
    **Insight:**  
    - Younger populations may show higher **social or property crimes**, often linked to social behavior and mobility.  
    - Older groups may have higher **white-collar** offenses due to occupational exposure.
    """)

    st.markdown("---")

    # === EDUCATION VS CRIME (VIOLIN PLOT) ===
    st.subheader("3️⃣ Education Level and Crime Type")

    education_cols = ['high_school_below', 'high_school', 'some_college', 'bachelors_degree']
    crime_cols = ['violent_crime', 'property_crime', 'whitecollar_crime', 'social_crime']

    crime_melted = df_uber_cleaned.melt(
        value_vars=crime_cols,
        var_name='Crime Type',
        value_name='Crime Score',
        id_vars=education_cols
    )

    education_crime_melted = crime_melted.melt(
        id_vars=['Crime Type', 'Crime Score'],
        value_vars=education_cols,
        var_name='Education Level',
        value_name='Education Percentage'
    )

    fig3, ax3 = plt.subplots(figsize=(12, 6))
    sns.violinplot(
        x='Education Level', y='Crime Score', hue='Crime Type',
        data=education_crime_melted, split=True, inner='quartile', palette='viridis', ax=ax3
    )
    ax3.set_title('Distribution of Crime Scores by Education Level and Crime Type', fontsize=13)
    ax3.set_xlabel('Education Level')
    ax3.set_ylabel('Crime Score')
    plt.xticks(rotation=45, ha='right')
    ax3.legend(title='Crime Type')
    st.pyplot(fig3)

    st.info("""
    **Insight:**  
    - **Higher education levels** are often associated with reduced **violent and property crimes**, 
      but can correlate with higher **white-collar offenses**.  
    - This reflects how **education access** influences both opportunity and crime typology.
    """)

    st.markdown("---")
    st.success("""
    ✅ **Summary:**  
    This demographic-focused analysis complements the clustering results from *crime1.py*, 
    offering a deeper understanding of **who** (by gender, age, education) is most associated with 
    specific crime trends in urban environments.
    """)

else:
    st.error("Failed to load dataset. Please check the data source.")
