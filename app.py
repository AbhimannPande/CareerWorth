import streamlit as st
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor, Pool
import joblib
from pathlib import Path
import datetime
import smtplib
from email.message import EmailMessage
import plotly.graph_objects as go

# ======================
# CONSTANTS & CONFIGURATION
# ======================
MODEL_DIR = Path("models")
MAX_AGE = 80
MAX_HOURS = 100
PREDICTION_LOG = Path("prediction_log.csv")

COUNTRY_DATA = {
    'USA': {'symbol': '$', 'currency': 'USD', 'color': '#2563eb'},
    'India': {'symbol': '₹', 'currency': 'INR', 'color': '#ea580c'},
    'UK': {'symbol': '£', 'currency': 'GBP', 'color': '#16a34a'},
    'Canada': {'symbol': 'CA$', 'currency': 'CAD', 'color': '#9333ea'},
    'Germany': {'symbol': '€', 'currency': 'EUR', 'color': '#dc2626'},
    'Australia': {'symbol': 'AU$', 'currency': 'AUD', 'color': '#ca8a04'}
}

OCCUPATIONS = sorted([
    'Prof-specialty', 'Exec-managerial', 'Sales', 'Tech-support', 'Craft-repair',
    'Transport-moving', 'Adm-clerical', 'Farming-fishing', 'Handlers-cleaners',
    'Machine-op-inspct', 'Other-service', 'Protective-serv', 'Armed-Forces',
    'IT', 'Healthcare', 'Legal'
])

EDUCATION_LEVELS = sorted([
    "HS-grad", "Some-college", "Assoc-voc", "Assoc-acdm", 
    "Bachelors", "Masters", "Prof-school", "Doctorate"
])

# ======================
# MODEL LOADING
# ======================
@st.cache_resource
def load_model():
    model = CatBoostRegressor()
    model.load_model(MODEL_DIR / "salary_model.cbm")
    feature_info = joblib.load(MODEL_DIR / "feature_info.joblib")
    return model, feature_info

model, feature_info = load_model()

# ======================
# STREAMLIT UI
# ======================
st.set_page_config(
    page_title="💼 CareerWorth",
    layout="wide",
    page_icon="💼",
    initial_sidebar_state="expanded"
)

# Stylish header with gradient and tagline
st.markdown("""
<div style="background: linear-gradient(to right, #2563eb, #1e40af); 
            padding: 1.2rem 2rem; border-radius: 12px; margin-bottom: 2rem;">
    <h1 style="color: white; margin: 0; font-size: 2.5rem;">
        💼 CareerWorth
    </h1>
    <p style="color: #dbeafe; margin: 0; font-size: 1.1rem;">
        Unlock your earning potential, globally.
    </p>
</div>
""", unsafe_allow_html=True)


# Sidebar
with st.sidebar:
    st.title("📊 Salary Insights")
    st.markdown("""
    **Welcome to CareerWorth**  
    This advanced tool helps you estimate earning potential based on:
    
    - Professional background
    - Education level
    - Work patterns
    - Geographic factors

    **Key Features:**
    - Age-adjusted salary predictions
    - Workload impact analysis
    - Country-specific benchmarks
    - Senior worker considerations

    
    """)
    st.divider()
    st.markdown("""
    **For optimal results:**
    1. Provide accurate occupation details
    2. Select closest education level
    3. Consider typical work hours
    """)
    st.divider()
    st.caption("© 2025 CareerWorth | v2.1.0")

# ================
# Tabs: Input, Download, Feedback
# ================
tabs = st.tabs(["📝 Input Form", "📁 Download Report", "📬 Feedback"])

# Globals for prediction results
prediction_result = None
input_summary = None

# =================
# Tab 1: Input Form + Prediction + Chart
# =================
with tabs[0]:
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)

        with col1:
            age = st.slider("Age", 18, MAX_AGE, 35, help="Select your current age")
            country = st.selectbox("Country", list(COUNTRY_DATA.keys()),
                                   format_func=lambda x: f"{COUNTRY_DATA[x]['symbol']} {x}")
            occupation = st.selectbox("Occupation", OCCUPATIONS, help="Select your primary occupation")
            education = st.selectbox("Education Level", EDUCATION_LEVELS,
                                     index=EDUCATION_LEVELS.index("Bachelors"))

        with col2:
            marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
            relationship = st.selectbox("Relationship", ["Husband", "Wife", "Unmarried"])
            gender = st.selectbox("Gender", ["Male", "Female", "Other"])
            hours_per_week = st.slider("Weekly Work Hours", 10, MAX_HOURS, 40,
                                       help="Typical hours worked per week")
            capital_gain = st.number_input("Capital Gains", min_value=0, value=0)
            capital_loss = st.number_input("Capital Losses", min_value=0, value=0)

        submitted = st.form_submit_button("Calculate Salary", type="primary")

    # Senior work warning
    if submitted and age > 60 and hours_per_week > 40:
        st.warning("""
        ⚠️ Notice for Senior Professionals:
        Working more than 40 hours/week after age 60 may impact:
        - Work-life balance
        - Health considerations
        - Retirement planning
        """)

    # Prediction Logic
    if submitted:
        input_data = {
            "age": age,
            "workclass": "Private",
            "education": education,
            "marital-status": marital_status,
            "occupation": occupation,
            "relationship": relationship,
            "gender": gender,
            "native-country": country,
            "hours-per-week": hours_per_week,
            "capital-gain": capital_gain,
            "capital-loss": capital_loss
        }

        try:
            input_df = pd.DataFrame([input_data])
            input_df['total_capital'] = input_df['capital-gain'] - input_df['capital-loss']
            input_df['log_capital-gain'] = np.log1p(input_df['capital-gain'])
            input_df['log_capital-loss'] = np.log1p(input_df['capital-loss'])

            for feature in feature_info['feature_order']:
                if feature not in input_df.columns:
                    input_df[feature] = 0

            input_df = input_df[feature_info['feature_order']]

            pool = Pool(input_df, cat_features=feature_info['categorical_cols'])
            local_salary = np.expm1(model.predict(pool)[0])
            country_info = COUNTRY_DATA[country]

            prediction_result = local_salary
            input_summary = input_data.copy()
            input_summary['Predicted Salary'] = local_salary
            input_summary['Timestamp'] = datetime.datetime.now().isoformat()

            # Save prediction to CSV log
            log_df = pd.DataFrame([input_summary])
            if PREDICTION_LOG.exists():
                log_df.to_csv(PREDICTION_LOG, mode='a', header=False, index=False)
            else:
                log_df.to_csv(PREDICTION_LOG, index=False)

            # Blue-styled salary display
            st.markdown(f"""
    ### 📊 Prediction Results
    <div style="padding: 1.5rem; background: #e0f2fe; border-radius: 10px; border-left: 8px solid #2563eb;">
        <div style="font-size: 2rem; font-weight: bold; color: #1d4ed8;">
            {country_info['symbol']}{local_salary:,.2f} {country_info['currency']}/year
        </div>
        <div style="margin-top: 1rem; display: flex; gap: 2rem; font-size: 1.1rem; color: #1e3a8a;">
            <span>≈ {country_info['symbol']}{local_salary/12:,.2f}/month</span>
            <span>≈ {country_info['symbol']}{local_salary/52:,.2f}/week</span>
            <span>≈ {country_info['symbol']}{local_salary/(52*hours_per_week):,.2f}/hour</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

            # Salary comparison chart
            import plotly.graph_objects as go
            chart_data = {
                "Your Salary": local_salary,
                f"{country} Avg": COUNTRY_DATA[country]['symbol'] + " ~" + "{:,.2f}".format(0)  # placeholder, we can use dummy average
            }
            # For demo, just use fixed country averages (hardcode or random)
            COUNTRY_AVERAGES = {
                'USA': 85000,
                'India': 600000,
                'UK': 42000,
                'Canada': 70000,
                'Germany': 60000,
                'Australia': 75000
            }
            bar_labels = ["Your Salary", f"{country} Avg"]
            bar_values = [local_salary, COUNTRY_AVERAGES.get(country, 0)]

            fig = go.Figure(data=[go.Bar(
                x=bar_labels,
                y=bar_values,
                marker_color=['#2563eb', '#93c5fd']
            )])
            fig.update_layout(
                title="Salary Comparison",
                yaxis_title=f"Annual Salary ({country_info['currency']})",
                xaxis_title="Category",
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
            )
            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Prediction error: {str(e)}")

# =================
# Tab 2: Download Report
# =================
with tabs[1]:
    st.header("📁 Download Prediction History")
    if PREDICTION_LOG.exists():
        df_log = pd.read_csv(PREDICTION_LOG)
        st.dataframe(df_log)
        csv_data = df_log.to_csv(index=False).encode("utf-8")
        st.download_button("Download CSV", csv_data, "careerworth_prediction_log.csv", "text/csv")
    else:
        st.info("No prediction history found yet.")

# =================
# Tab 3: Feedback
# =================
with tabs[2]:
    st.header("💬 Send Feedback")
    feedback = st.text_area("What do you think about CareerWorth?")
    user_email = st.text_input("Your Email (optional)")

    if st.button("Submit Feedback"):
        if not feedback.strip():
            st.warning("Please enter some feedback before submitting.")
        else:
            # Send email logic here (replace with your real creds)
            try:
                msg = EmailMessage()
                msg.set_content(f"Feedback:\n{feedback}\n\nFrom: {user_email if user_email else 'Anonymous'}")
                msg['Subject'] = 'CareerWorth Feedback'
                msg['From'] = "your_email@gmail.com"  # Replace with your sender email
                msg['To'] = "p0851216@gmail.com"       # Your receiving email

                # SMTP connection (Gmail example)
                with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
                    smtp.login("your_email@gmail.com", "your_app_password")  # Replace creds
                    smtp.send_message(msg)

                st.success("Thanks for your feedback! 🙌")
            except Exception as e:
                st.error(f"Error sending feedback: {e}")

# ======================
# Footer with Social Links
# ======================
st.markdown("---")
st.markdown("### 📬 Connect with the Developer", unsafe_allow_html=True)

footer_html = """
<div style='text-align: center; margin-top: 1rem;'>
    <a href='https://www.linkedin.com/in/abhimann-pande-17791a2a6' target='_blank' style='margin: 0 15px; text-decoration: none;'>
        <img src='https://cdn.jsdelivr.net/gh/devicons/devicon/icons/linkedin/linkedin-original.svg' width='30' />
    </a>
    <a href='https://github.com/AbhimannPande' target='_blank' style='margin: 0 15px; text-decoration: none;'>
        <img src='https://cdn.jsdelivr.net/gh/devicons/devicon/icons/github/github-original.svg' width='30' />
    </a>
    <a href='mailto:p0851216@gmail.com' style='margin: 0 15px; text-decoration: none;'>
        <img src='https://img.icons8.com/fluency/48/gmail-new.png' width='30' />
    </a>
    <p style='margin-top: 0.5rem; font-size: 0.9rem;'>Made by Abhimann Pande</p>
</div>
"""
st.markdown(footer_html, unsafe_allow_html=True)
