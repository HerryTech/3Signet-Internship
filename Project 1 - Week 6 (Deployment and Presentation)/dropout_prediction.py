import joblib
import streamlit as st

# Load the model using joblib
model = joblib.load("C:/Users/USER/Desktop/3Signet/Project 1 - Week 6 (Deployment and Presentation)/final_model.pkl")

def main():
    st.title("School Dropout Prediction")

    # Input variables
    Previous_qualification = st.text_input("Previous qualification (grade)")
    Debtor = st.text_input("Debtor")
    Tuition_fees_UpToDate = st.text_input("Tuition fees up to date")
    Educational_special_needs = st.text_input("Educational special needs")
    Scholarship_holder = st.text_input("Scholarship holder")
    International = st.text_input("International")
    Curricular_units_1st_sem_credited = st.text_input("Curricular units 1st sem (credited)")
    Curricular_units_1st_sem_enrolled = st.text_input("Curricular units 1st sem (enrolled)")
    Curricular_units_1st_sem_evaluations = st.text_input("Curricular units 1st sem (evaluations)")
    Curricular_units_1st_sem_approved = st.text_input("Curricular units 1st sem (approved)")
    Curricular_units_1st_sem_without_evaluations = st.text_input("Curricular units 1st sem (without evaluations)")
    Curricular_units_1st_sem_grade = st.text_input("Curricular units 1st sem (grade)")
    Curricular_units_2nd_sem_credited = st.text_input("Curricular units 2nd sem (credited)")
    Curricular_units_2nd_sem_enrolled = st.text_input("Curricular units 2nd sem (enrolled)")
    Curricular_units_2nd_sem_evaluations = st.text_input("Curricular units 2nd sem (evaluations)")
    Curricular_units_2nd_sem_approved = st.text_input("Curricular units 2nd sem (approved)")
    Curricular_units_2nd_sem_without_evaluations = st.text_input("Curricular units 2nd sem (without evaluations)")
    Curricular_units_2nd_sem_grade = st.text_input("Curricular units 2nd sem (grade)")
    Age_at_enrollment = st.text_input("Age at enrollment")

    # Check if any input is empty
    inputs = [
        Previous_qualification, Debtor, Tuition_fees_UpToDate,
        Educational_special_needs, Scholarship_holder, International,
        Curricular_units_1st_sem_credited, Curricular_units_1st_sem_enrolled,
        Curricular_units_1st_sem_evaluations, Curricular_units_1st_sem_approved,
        Curricular_units_1st_sem_without_evaluations, Curricular_units_1st_sem_grade,
        Curricular_units_2nd_sem_credited, Curricular_units_2nd_sem_enrolled,
        Curricular_units_2nd_sem_evaluations, Curricular_units_2nd_sem_approved,
        Curricular_units_2nd_sem_without_evaluations, Curricular_units_2nd_sem_grade,
        Age_at_enrollment
    ]

    if st.button("Predict Dropout"):
        if any(input == '' for input in inputs):
            st.error("Please fill in all the fields.")
        else:
            # Convert inputs to the appropriate types
            make_prediction = model.predict([[float(Previous_qualification), float(Debtor), float(Tuition_fees_UpToDate),
            Educational_special_needs, Scholarship_holder, International,
            float(Curricular_units_1st_sem_credited), float(Curricular_units_1st_sem_enrolled),
            float(Curricular_units_1st_sem_evaluations), float(Curricular_units_1st_sem_approved),
            float(Curricular_units_1st_sem_without_evaluations), float(Curricular_units_1st_sem_grade),
            float(Curricular_units_2nd_sem_credited), float(Curricular_units_2nd_sem_enrolled),
            float(Curricular_units_2nd_sem_evaluations), float(Curricular_units_2nd_sem_approved),
            float(Curricular_units_2nd_sem_without_evaluations), float(Curricular_units_2nd_sem_grade),
            float(Age_at_enrollment)]])

            if make_prediction == 0:
                result = "The student will drop out"
            elif make_prediction == 1:
                result = "The student enrolled"
            else:
                result = "The student will graduate"
            
            st.success(result)

if __name__ == '__main__':
    main()
