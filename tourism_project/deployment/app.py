import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download and load the trained model
model_path = hf_hub_download(repo_id="irenerini/tourism_package_model", filename="best_tourism_package_model_v1.joblib")
model = joblib.load(model_path)

# Streamlit UI
st.title("Tourism Package Prediction")
st.write("""
This application predicts whether a customer will purchase the newly introduced Wellness Tourism Package before contacting them.
Please enter the app details below to get a revenue prediction.
""")

# Input fields for user data
with st.form("prediction_form"):
    st.header("Customer Details")
    col1, col2 = st.columns(2)

    with col1:
        Age = st.number_input("Age", min_value=18, max_value=90, value=30)
        TypeofContact_options = {"Company Invited": 0, "Self Inquiry": 1}
        TypeofContact_display = st.selectbox("Type of Contact", list(TypeofContact_options.keys()))
        TypeofContact = TypeofContact_options[TypeofContact_display]
        CityTier = st.selectbox("City Tier", [1, 2, 3])
        DurationOfPitch = st.number_input("Duration of Pitch (minutes)", min_value=1.0, max_value=60.0, value=10.0)
        Occupation_options = {"Free Lancer": 0, "Salaried": 1, "Small Business": 2, "Large Business": 3}
        Occupation_display = st.selectbox("Occupation", list(Occupation_options.keys()))
        Occupation = Occupation_options[Occupation_display]
        Gender_options = {"Male": 1, "Female": 0}
        Gender_display = st.selectbox("Gender", list(Gender_options.keys()))
        Gender = Gender_options[Gender_display]
        NumberOfPersonVisiting = st.number_input("Number of Persons Visiting", min_value=1, max_value=10, value=2)
        NumberOfFollowups = st.number_input("Number of Follow-ups", min_value=0, max_value=20, value=3)

    with col2:
        ProductPitched_options = {"Basic": 0, "Standard": 1, "Kind": 2, "Deluxe": 3, "Super Deluxe": 4}
        ProductPitched_display = st.selectbox("Product Pitched", list(ProductPitched_options.keys()))
        ProductPitched = ProductPitched_options[ProductPitched_display]
        PreferredPropertyStar = st.selectbox("Preferred Property Star", [3, 4, 5])
        MaritalStatus_options = {"Single": 2, "Divorced": 0, "Married": 1, "Unmarried": 3}
        MaritalStatus_display = st.selectbox("Marital Status", list(MaritalStatus_options.keys()))
        MaritalStatus = MaritalStatus_options[MaritalStatus_display]
        NumberOfTrips = st.number_input("Number of Trips Annually", min_value=0, max_value=50, value=1)
        Passport = st.selectbox("Passport", [0, 1])
        PitchSatisfactionScore = st.selectbox("Pitch Satisfaction Score", [1, 2, 3, 4, 5])
        OwnCar = st.selectbox("Own Car", [0, 1])
        NumberOfChildrenVisiting = st.number_input("Number of Children Visiting (below 5)", min_value=0, max_value=10, value=0)
        Designation_options = {"Senior Manager": 4, "Manager": 2, "Executive": 1, "AVP": 0, "VP": 5}
        Designation_display = st.selectbox("Designation", list(Designation_options.keys()))
        Designation = Designation_options[Designation_display]
        MonthlyIncome = st.number_input("Monthly Income", min_value=0.0, value=20000.0)

    submitted = st.form_submit_button("Predict Purchase")

    if submitted:
        # Create a DataFrame from the input data
        input_data = pd.DataFrame([[Age, TypeofContact, CityTier, DurationOfPitch, Occupation, Gender,
                                      NumberOfPersonVisiting, NumberOfFollowups, ProductPitched, PreferredPropertyStar,
                                      MaritalStatus, NumberOfTrips, Passport, PitchSatisfactionScore, OwnCar,
                                      NumberOfChildrenVisiting, Designation, MonthlyIncome]],
                                    columns=['Age', 'TypeofContact', 'CityTier', 'DurationOfPitch', 'Occupation', 'Gender',
                                             'NumberOfPersonVisiting', 'NumberOfFollowups', 'ProductPitched', 'PreferredPropertyStar',
                                             'MaritalStatus', 'NumberOfTrips', 'Passport', 'PitchSatisfactionScore', 'OwnCar',
                                             'NumberOfChildrenVisiting', 'Designation', 'MonthlyIncome'])

        # Make prediction
        prediction = model.predict(input_data)[0]

        st.subheader("Prediction Result")
        if prediction > 0.5: # Assuming a threshold for binary classification
            st.success(f"The customer is likely to purchase the package (Prediction Score: {prediction:.2f})")
        else:
            st.info(f"The customer is unlikely to purchase the package (Prediction Score: {prediction:.2f})")
