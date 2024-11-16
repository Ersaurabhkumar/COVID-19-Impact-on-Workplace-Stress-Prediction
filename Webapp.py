import streamlit as st
import joblib
import numpy as np

# Load model and individual encoders
model = joblib.load('model.pkl')
label_encoders = joblib.load('label_encoders.pkl')  # This is a dictionary of encoders
scaler = joblib.load('scaler.pkl')

# Title and inputs
st.title('Predict Stress Level')
st.write('Enter the details to predict mental health condition')

# Inputs for each feature, accessing the encoder for each column individually
Sector = st.selectbox('Sector', label_encoders['Sector'].classes_)
Increased_Work_Hours = st.number_input('Increased_Work_Hours', min_value=0, max_value=65, step=1)
Health_Issue = st.number_input('Health_Issue', min_value=0, max_value=1, step=1)
Work_From_Home = st.number_input('Work_From_Home')
Hours_Worked_Per_Day = st.number_input('Hours_Worked_Per_Day')
Meetings_Per_Day = st.number_input('Meetings_Per_Day', min_value=0, max_value=10, step=1)
Productivity_Change = st.number_input('Productivity_Change', min_value=0, max_value=10, step=1)
Job_Security = st.number_input('Job_Security', min_value=0, max_value=10, step=1)
Commuting_Changes = st.number_input('Commuting_Changes', min_value=0, max_value=10, step=1)
Technology_Adaptation = st.number_input('Technology_Adaptation', min_value=0, max_value=10, step=1)
Salary_Changes = st.number_input('Salary_Changes', min_value=0, max_value=10, step=1)
Team_Collaboration_Challenges = st.number_input('Team_Collaboration_Challenges', min_value=0, max_value=10, step=1)
Affected_by_Covid = st.number_input('Affected_by_Covid', min_value=0, max_value=10, step=1)
Total_Work_From_Home_Time = st.number_input('Total_Work_From_Home_Time', min_value=0, max_value=10, step=1)
Childcare_Responsibilities = st.number_input('Childcare_Responsibilities', min_value=0, max_value=10, step=1)

# Encode categorical inputs
encoded_inputs = np.array([
    label_encoders['Sector'].transform([Sector])[0]
])

# Numerical inputs (no encoding, just numerical features)
numerical_inputs = np.array([
    Increased_Work_Hours, Health_Issue, Work_From_Home,
    Hours_Worked_Per_Day, Meetings_Per_Day, Productivity_Change,
    Job_Security, Commuting_Changes, Technology_Adaptation,
    Salary_Changes, Team_Collaboration_Challenges, Affected_by_Covid,
    Total_Work_From_Home_Time, Childcare_Responsibilities
])

# Combine encoded categorical and numerical inputs into a single array
combined_inputs = np.hstack((encoded_inputs, numerical_inputs)).reshape(1, -1)

# Scale the combined input features
scaled_inputs = scaler.transform(combined_inputs)

prediction = None

# Predict Stress Level when the button is clicked
if st.button("Predict Stress Level"):
    prediction = model.predict(scaled_inputs)
    st.write(f"Stress Level: {prediction[0]}")

# Check the prediction value only if it has been set
if prediction is not None:
    if prediction == 0:
        st.write('The person is Highly stressed')
    elif prediction == 1:
        st.write('The person is Low Stressed')
    elif prediction == 2:
        st.write('The person is Medium Stressed')
    else:
        st.write('Unknown stress level')

