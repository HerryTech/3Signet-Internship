import joblib
import streamlit as st

# Load the model using joblib
model = joblib.load("C:/Users/USER/Desktop/3Signet/Project 1 - Week 6 (Deployment and Presentation)/final_model.pkl")

def main():
    st.title("School Dropout Prediction")

    # Initialize session state for inputs
    if 'inputs' not in st.session_state:
        st.session_state.inputs = {
            'Previous_qualification': '',
            'Debtor': '',
            'Tuition_fees_UpToDate': '',
            'Educational_special_needs': '',
            'Scholarship_holder': '',
            'International': '',
            'Curricular_units_1st_sem_credited': '',
            'Curricular_units_1st_sem_enrolled': '',
            'Curricular_units_1st_sem_evaluations': '',
            'Curricular_units_1st_sem_approved': '',
            'Curricular_units_1st_sem_without_evaluations': '',
            'Curricular_units_1st_sem_grade': '',
            'Curricular_units_2nd_sem_credited': '',
            'Curricular_units_2nd_sem_enrolled': '',
            'Curricular_units_2nd_sem_evaluations': '',
            'Curricular_units_2nd_sem_approved': '',
            'Curricular_units_2nd_sem_without_evaluations': '',
            'Curricular_units_2nd_sem_grade': '',
            'Age_at_enrollment': ''
        }

    # Input variables
    for key in st.session_state.inputs:
        st.session_state.inputs[key] = st.text_input(key.replace('_', ' ').capitalize(), value=st.session_state.inputs[key])

    if st.button("Predict"):
        # Check for empty inputs
        if any(value == '' for value in st.session_state.inputs.values()):
            st.error("Please fill in all the fields.")
        else:
            try:
                # Convert inputs to the appropriate types
                make_prediction = model.predict([[float(st.session_state.inputs['Previous_qualification']), 
                                                   float(st.session_state.inputs['Debtor']), 
                                                   float(st.session_state.inputs['Tuition_fees_UpToDate']),
                                                   st.session_state.inputs['Educational_special_needs'], 
                                                   st.session_state.inputs['Scholarship_holder'], 
                                                   st.session_state.inputs['International'],
                                                   float(st.session_state.inputs['Curricular_units_1st_sem_credited']), 
                                                   float(st.session_state.inputs['Curricular_units_1st_sem_enrolled']),
                                                   float(st.session_state.inputs['Curricular_units_1st_sem_evaluations']), 
                                                   float(st.session_state.inputs['Curricular_units_1st_sem_approved']),
                                                   float(st.session_state.inputs['Curricular_units_1st_sem_without_evaluations']), 
                                                   float(st.session_state.inputs['Curricular_units_1st_sem_grade']),
                                                   float(st.session_state.inputs['Curricular_units_2nd_sem_credited']), 
                                                   float(st.session_state.inputs['Curricular_units_2nd_sem_enrolled']),
                                                   float(st.session_state.inputs['Curricular_units_2nd_sem_evaluations']), 
                                                   float(st.session_state.inputs['Curricular_units_2nd_sem_approved']),
                                                   float(st.session_state.inputs['Curricular_units_2nd_sem_without_evaluations']), 
                                                   float(st.session_state.inputs['Curricular_units_2nd_sem_grade']),
                                                   float(st.session_state.inputs['Age_at_enrollment'])]])

                if make_prediction == 0:
                    result = "The student will drop out"
                elif make_prediction == 1:
                    result = "The student enrolled"
                else:
                    result = "The student will graduate"

                st.success(result)

                # Clear inputs after prediction
                for key in st.session_state.inputs:
                    st.session_state.inputs[key] = ''
            except ValueError:
                st.error("Please input numeric values only.")

if __name__ == '__main__':
    main()
