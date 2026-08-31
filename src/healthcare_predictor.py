import pandas as pd

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split


FEATURES = [

    "Age",

    "BMI",

    "Blood_Pressure",

    "Glucose",

    "Cholesterol",

    "Physical_Activity_Days",

    "Smoking",

    "Family_History"
]


def train_healthcare_model(data):

    X = data[
        FEATURES
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


    return model


def predict_risk(
    patient_data,
    model
):

    patient_df = pd.DataFrame(

        [patient_data]
    )


    prediction = model.predict(

        patient_df[
            FEATURES
        ]
    )[0]


    probabilities = model.predict_proba(

        patient_df[
            FEATURES
        ]
    )[0]


    risk_mapping = {

        0: "Low",

        1: "Moderate",

        2: "High"
    }


    return {

        "Risk_Level": risk_mapping[
            prediction
        ],

        "Confidence": float(

            probabilities[
                prediction
            ]
        )
    }
