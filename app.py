import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# -------------------------------------------------
# PAGE CONFIGURATION
# -------------------------------------------------

st.set_page_config(
    page_title="AI Healthcare Risk Prediction",
    page_icon="🏥",
    layout="wide"
)


# -------------------------------------------------
# CREATE DEMONSTRATION DATASET
# -------------------------------------------------

@st.cache_data
def create_healthcare_dataset():

    np.random.seed(42)

    samples = 2000


    age = np.random.randint(
        18,
        90,
        samples
    )


    bmi = np.random.normal(
        26,
        5,
        samples
    )


    bmi = np.clip(
        bmi,
        15,
        50
    )


    blood_pressure = np.random.normal(
        120,
        18,
        samples
    )


    blood_pressure = np.clip(
        blood_pressure,
        80,
        200
    )


    glucose = np.random.normal(
        105,
        30,
        samples
    )


    glucose = np.clip(
        glucose,
        60,
        300
    )


    cholesterol = np.random.normal(
        190,
        40,
        samples
    )


    cholesterol = np.clip(
        cholesterol,
        100,
        400
    )


    physical_activity = np.random.randint(
        0,
        8,
        samples
    )


    smoking = np.random.choice(
        [0, 1],
        samples,
        p=[0.75, 0.25]
    )


    family_history = np.random.choice(
        [0, 1],
        samples,
        p=[0.65, 0.35]
    )


    risk_score = (

        (age > 60).astype(int)

        +

        (bmi > 30).astype(int)

        +

        (blood_pressure > 140).astype(int)

        +

        (glucose > 140).astype(int)

        +

        (cholesterol > 240).astype(int)

        +

        (physical_activity < 2).astype(int)

        +

        smoking

        +

        family_history
    )


    risk = np.where(

        risk_score <= 2,

        0,

        np.where(

            risk_score <= 4,

            1,

            2
        )
    )


    data = pd.DataFrame({

        "Age": age,

        "BMI": bmi.round(1),

        "Blood_Pressure": blood_pressure.round(1),

        "Glucose": glucose.round(1),

        "Cholesterol": cholesterol.round(1),

        "Physical_Activity_Days": physical_activity,

        "Smoking": smoking,

        "Family_History": family_history,

        "Risk_Level": risk
    })


    return data


health_data = create_healthcare_dataset()


# -------------------------------------------------
# FEATURES
# -------------------------------------------------

features = [

    "Age",

    "BMI",

    "Blood_Pressure",

    "Glucose",

    "Cholesterol",

    "Physical_Activity_Days",

    "Smoking",

    "Family_History"
]


# -------------------------------------------------
# TRAIN MODEL
# -------------------------------------------------

@st.cache_resource
def train_model(data):

    X = data[
        features
    ]


    y = data[
        "Risk_Level"
    ]


    X_train, X_test, y_train, y_test = train_test_split(

        X,

        y,

        test_size=0.2,

        random_state=42,

        stratify=y
    )


    model = RandomForestClassifier(

        n_estimators=200,

        random_state=42
    )


    model.fit(
        X_train,
        y_train
    )


    predictions = model.predict(
        X_test
    )


    accuracy = accuracy_score(
        y_test,
        predictions
    )


    return model, accuracy


model, model_accuracy = train_model(
    health_data
)


# -------------------------------------------------
# TITLE
# -------------------------------------------------

st.title(
    "🏥 AI Healthcare Risk Prediction & Analytics"
)


st.markdown(
    "An educational Machine Learning project that analyzes "
    "health-related indicators and estimates demonstration "
    "risk levels using predictive analytics."
)


# -------------------------------------------------
# IMPORTANT DISCLAIMER
# -------------------------------------------------

st.warning(
    "⚠️ Educational Demonstration Only: "
    "This application is a portfolio Machine Learning project "
    "using synthetic demonstration data. It is NOT a medical "
    "diagnosis tool and should not be used for medical decisions."
)


# -------------------------------------------------
# SIDEBAR
# -------------------------------------------------

st.sidebar.header(
    "🧠 How the System Works"
)


st.sidebar.info(
    """
    1. Health indicators are provided.

    2. Features are processed.

    3. A Machine Learning model analyzes patterns.

    4. The model estimates a demonstration risk level.

    5. Analytics and feature importance are displayed.
    """
)


# -------------------------------------------------
# DASHBOARD METRICS
# -------------------------------------------------

st.subheader(
    "📊 Healthcare Dataset Overview"
)


total_records = len(
    health_data
)


low_risk = len(

    health_data[
        health_data[
            "Risk_Level"
        ] == 0
    ]
)


moderate_risk = len(

    health_data[
        health_data[
            "Risk_Level"
        ] == 1
    ]
)


high_risk = len(

    health_data[
        health_data[
            "Risk_Level"
        ] == 2
    ]
)


col1, col2, col3, col4 = st.columns(4)


col1.metric(
    "Total Records",
    total_records
)


col2.metric(
    "Low Risk",
    low_risk
)


col3.metric(
    "Moderate Risk",
    moderate_risk
)


col4.metric(
    "High Risk",
    high_risk
)


# -------------------------------------------------
# DATASET PREVIEW
# -------------------------------------------------

st.subheader(
    "📁 Demonstration Healthcare Dataset"
)


display_data = health_data.copy()


display_data[
    "Risk_Category"
] = display_data[
    "Risk_Level"
].map({

    0: "Low",

    1: "Moderate",

    2: "High"
})


st.dataframe(
    display_data.head(30),
    use_container_width=True
)


# -------------------------------------------------
# RISK DISTRIBUTION
# -------------------------------------------------

st.subheader(
    "📊 Risk Level Distribution"
)


risk_counts = (

    display_data[
        "Risk_Category"
    ]
    .value_counts()
    .reset_index()
)


risk_counts.columns = [

    "Risk Level",

    "Count"
]


fig_risk = px.pie(

    risk_counts,

    names="Risk Level",

    values="Count",

    title="Demonstration Risk Distribution"
)


st.plotly_chart(
    fig_risk,
    use_container_width=True
)


# -------------------------------------------------
# HEALTH ANALYTICS
# -------------------------------------------------

st.subheader(
    "📈 Healthcare Indicator Analysis"
)


fig_health = px.scatter(

    display_data,

    x="BMI",

    y="Glucose",

    color="Risk_Category",

    size="Age",

    hover_data=[

        "Blood_Pressure",

        "Cholesterol"
    ],

    title="BMI vs Glucose Analysis"
)


st.plotly_chart(
    fig_health,
    use_container_width=True
)


# -------------------------------------------------
# RISK PREDICTION
# -------------------------------------------------

st.divider()


st.header(
    "🔍 Analyze Health Indicators"
)


col1, col2 = st.columns(2)


with col1:

    age = st.number_input(

        "Age",

        min_value=18,

        max_value=100,

        value=30
    )


    bmi = st.number_input(

        "BMI",

        min_value=10.0,

        max_value=60.0,

        value=24.0
    )


    blood_pressure = st.number_input(

        "Blood Pressure",

        min_value=70.0,

        max_value=250.0,

        value=120.0
    )


    glucose = st.number_input(

        "Glucose Level",

        min_value=40.0,

        max_value=400.0,

        value=100.0
    )


with col2:

    cholesterol = st.number_input(

        "Cholesterol Level",

        min_value=80.0,

        max_value=500.0,

        value=190.0
    )


    physical_activity = st.slider(

        "Physical Activity Days per Week",

        min_value=0,

        max_value=7,

        value=3
    )


    smoking = st.selectbox(

        "Smoking",

        [

            "No",

            "Yes"
        ]
    )


    family_history = st.selectbox(

        "Family History",

        [

            "No",

            "Yes"
        ]
    )


# -------------------------------------------------
# PREDICTION
# -------------------------------------------------

if st.button(
    "🤖 Estimate Demonstration Risk"
):

    smoking_value = (

        1

        if smoking == "Yes"

        else 0
    )


    family_history_value = (

        1

        if family_history == "Yes"

        else 0
    )


    input_data = pd.DataFrame({

        "Age": [
            age
        ],

        "BMI": [
            bmi
        ],

        "Blood_Pressure": [
            blood_pressure
        ],

        "Glucose": [
            glucose
        ],

        "Cholesterol": [
            cholesterol
        ],

        "Physical_Activity_Days": [
            physical_activity
        ],

        "Smoking": [
            smoking_value
        ],

        "Family_History": [
            family_history_value
        ]
    })


    prediction = model.predict(
        input_data
    )[0]


    probabilities = model.predict_proba(
        input_data
    )[0]


    risk_mapping = {

        0: "Low Demonstration Risk",

        1: "Moderate Demonstration Risk",

        2: "High Demonstration Risk"
    }


    risk_result = risk_mapping[
        prediction
    ]


    st.divider()


    st.subheader(
        "🤖 Machine Learning Result"
    )


    col1, col2 = st.columns(2)


    col1.metric(
        "Estimated Risk Level",
        risk_result
    )


    confidence = probabilities[
        prediction
    ] * 100


    col2.metric(
        "Model Confidence",
        f"{confidence:.1f}%"
    )


    if prediction == 0:

        st.success(
            "The demonstration model estimated a lower risk category."
        )


    elif prediction == 1:

        st.warning(
            "The demonstration model estimated a moderate risk category."
        )


    else:

        st.error(
            "The demonstration model estimated a higher risk category."
        )


    st.info(
        "This output is generated by a demonstration Machine Learning "
        "model using synthetic data and is not medical advice or diagnosis."
    )


# -------------------------------------------------
# FEATURE IMPORTANCE
# -------------------------------------------------

st.divider()


st.subheader(
    "🧠 Feature Importance"
)


feature_importance = pd.DataFrame({

    "Feature": features,

    "Importance": model.feature_importances_
})


feature_importance = feature_importance.sort_values(

    by="Importance",

    ascending=False
)


fig_importance = px.bar(

    feature_importance,

    x="Importance",

    y="Feature",

    orientation="h",

    title="Features Influencing the Demonstration Model"
)


st.plotly_chart(
    fig_importance,
    use_container_width=True
)


# -------------------------------------------------
# CSV UPLOAD
# -------------------------------------------------

st.divider()


st.header(
    "📤 Batch Data Analysis"
)


uploaded_file = st.file_uploader(

    "Upload a CSV containing the required indicators",

    type=["csv"]
)


if uploaded_file is not None:

    uploaded_data = pd.read_csv(
        uploaded_file
    )


    missing_columns = [

        feature

        for feature in features

        if feature not in uploaded_data.columns
    ]


    if missing_columns:

        st.error(
            "Missing required columns: "
            +
            ", ".join(
                missing_columns
            )
        )


    else:

        predictions = model.predict(

            uploaded_data[
                features
            ]
        )


        probabilities = model.predict_proba(

            uploaded_data[
                features
            ]
        )


        uploaded_data[
            "Estimated_Risk"
        ] = predictions


        uploaded_data[
            "Risk_Category"
        ] = uploaded_data[
            "Estimated_Risk"
        ].map({

            0: "Low",

            1: "Moderate",

            2: "High"
        })


        uploaded_data[
            "Model_Confidence"
        ] = probabilities.max(
            axis=1
        )


        st.subheader(
            "Batch Analysis Results"
        )


        st.dataframe(
            uploaded_data,
            use_container_width=True
        )


# -------------------------------------------------
# FOOTER
# -------------------------------------------------

st.divider()


st.caption(
    "AI Healthcare Risk Prediction & Analytics | "
    "Python • Machine Learning • Random Forest • "
    "Predictive Analytics • Data Visualization"
)
