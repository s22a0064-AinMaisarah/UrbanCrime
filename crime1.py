import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# --- CONFIGURATION ---
CSV_URL = 'https://raw.githubusercontent.com/s22a0064-AinMaisarah/EC2024/refs/heads/main/cleaned_student_survey.csv'
ENCODING_TYPE = 'cp1252'

# --- APP TITLE ---
st.title("🎓 Visual Data Insights: 4th Year Student Analysis")
st.markdown("This dashboard presents five meaningful visualizations derived from the cleaned student survey dataset.")
st.markdown("---")

col1, col2, col3, col4 = st.columns(4)

col1.metric(label="PLO 2", value=f"3.3", help="PLO 2: Cognitive Skill", border=True)
col1.metric(label="PLO 3", value=f"3.5", help="PLO 3: Digital Skill", border=True)
col1.metric(label="PLO 4", value=f"4.0", help="PLO 4: Interpersonal Skill", border=True)
col1.metric(label="PLO 5", value=f"4.3", help="PLO 5: Communication Skill", border=True)


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
    # Clean missing data
    missing_percentage = df.isnull().sum() / len(df) * 100
    cols_to_drop = missing_percentage[missing_percentage > 50].index
    df_cleaned = df.drop(columns=cols_to_drop)

    for col in df_cleaned.columns:
        if df_cleaned[col].isnull().sum() > 0:
            df_cleaned[col].fillna(df_cleaned[col].mode()[0], inplace=True)

    # Filter for 4th-year students
    if 'Bachelor  Academic Year in EU' in df_cleaned.columns:
        fourth_year_students = df_cleaned[df_cleaned['Bachelor  Academic Year in EU'] == '4th Year']

        if not fourth_year_students.empty:
            # Visualization 1: Gender Distribution (Pie Chart)
            st.subheader("1️⃣ Gender Distribution of 4th Year Students")
            gender_counts = fourth_year_students['Gender'].value_counts().reset_index()
            gender_counts.columns = ['Gender', 'Count']
            fig1 = px.pie(gender_counts, names='Gender', values='Count', title='Gender Distribution of 4th Year Students')
            st.plotly_chart(fig1, use_container_width=True)
            st.write("""
            **Interpretation:**  
            The majority gender among 4th-year students can be clearly seen from the chart. 
            This helps understand diversity within the academic year and may influence 
            how support programs or student engagement activities are designed.
            """)

            # Visualization 2: Scatter Plot (H.S.C GPA vs Satisfaction)
            st.subheader("2️⃣ Scatter Plot: H.S.C (GPA) vs Satisfaction (Q5)")
            gpa_col = 'H.S.C (GPA)'
            satisfaction_col = 'Q5 [To what extent your expectation was met?]'
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.scatterplot(data=fourth_year_students, x=gpa_col, y=satisfaction_col, ax=ax)
            ax.set_title(f'{gpa_col} vs {satisfaction_col}')
            st.pyplot(fig)
            st.write("""
            **Interpretation:**  
            The scatter plot shows the relationship between students' GPA and their satisfaction level.  
            There appears to be a mild positive trend — students with higher GPAs may report slightly higher satisfaction.  
            This could imply that academic performance influences how well students feel their expectations are met.
            """)

            # Visualization 3: Box Plot of GPA by Gender
            st.subheader("3️⃣ Box Plot: H.S.C (GPA) Distribution by Gender")
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.boxplot(data=fourth_year_students, x='Gender', y='H.S.C (GPA)', ax=ax)
            ax.set_title('H.S.C (GPA) by Gender')
            st.pyplot(fig)
            st.write("""
            **Interpretation:**  
            The box plot highlights the GPA distribution across genders.  
            Both genders appear to perform similarly, though one may show slightly higher median values.  
            This visualization helps detect outliers and differences in academic performance by gender.
            """)

            # Visualization 4: Histogram of Satisfaction
            st.subheader("4️⃣ Histogram: Satisfaction Level Distribution (Q5)")
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.histplot(data=fourth_year_students, x=satisfaction_col, bins=5, kde=True, ax=ax)
            ax.set_title('Distribution of Satisfaction Levels (Q5)')
            st.pyplot(fig)
            st.write("""
            **Interpretation:**  
            Most students reported satisfaction scores clustered in the middle to high range.  
            This suggests that expectations are generally met among 4th-year students, 
            though there is room for improvement for those less satisfied.
            """)

            # Visualization 5: Violin Plot of GPA by Gender
            st.subheader("5️⃣ Violin Plot: H.S.C (GPA) by Gender")
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.violinplot(data=fourth_year_students, x='Gender', y='H.S.C (GPA)', ax=ax)
            ax.set_title('H.S.C (GPA) by Gender (Violin Plot)')
            st.pyplot(fig)
            st.write("""
            **Interpretation:**  
            The violin plot provides a deeper view of GPA distributions for each gender.  
            It shows not only median and quartile ranges but also the density of scores.  
            This reveals that both genders have a fairly even GPA spread, indicating similar academic performance patterns.
            """)

        else:
            st.warning("No data found for 4th Year students.")
    else:
        st.error("Column 'Bachelor  Academic Year in EU' not found in dataset.")
else:
    st.error("Failed to load dataset from GitHub.")
