import streamlit as st
import mysql.connector
from datetime import datetime
import pandas as pd
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import LabelEncoder

# Database connection function
def create_connection():
    try:
        # First try to connect without specifying a database
        connection = mysql.connector.connect(
            host="127.0.0.1",
            user="root",
            password="722133",
            port=3390
        )
        return connection
    except mysql.connector.Error as e:
        st.error(f"Error connecting to MySQL Database: {e}")
        return None

# Create database if it doesn't exist
def create_database():
    conn = create_connection()
    if conn is not None:
        cursor = conn.cursor()
        try:
            cursor.execute("CREATE DATABASE IF NOT EXISTS my_new_data")
            conn.commit()
            st.success("Database 'my_new_data' created successfully!")
        except mysql.connector.Error as e:
            st.error(f"Error creating database: {e}")
        finally:
            cursor.close()
            conn.close()

# Connect to the specific database
def connect_to_database():
    try:
        connection = mysql.connector.connect(
            host="127.0.0.1",
            user="root",
            password="722133",
            database="my_new_data",
            port=3390
        )
        return connection
    except mysql.connector.Error as e:
        st.error(f"Error connecting to MySQL Database: {e}")
        return None

# Create tables if they don't exist
def create_tables():
    conn = connect_to_database()
    if conn is not None:
        cursor = conn.cursor()
        
        # Create patients table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS patients (
                patient_id INT AUTO_INCREMENT PRIMARY KEY,
                nhs_number VARCHAR(20) UNIQUE,
                patient_name VARCHAR(100),
                date_of_birth DATE,
                patient_address TEXT,
                medical_conditions TEXT
            )
        """)
        
        # Create prescriptions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS prescriptions (
                prescription_id INT AUTO_INCREMENT PRIMARY KEY,
                patient_id INT,
                name_of_tablets VARCHAR(100),
                reference_no VARCHAR(50),
                dose VARCHAR(50),
                number_of_tablets INT,
                lot VARCHAR(50),
                issue_date DATE,
                exp_date DATE,
                daily_dose VARCHAR(50),
                side_effect TEXT,
                further_information TEXT,
                storage_advice TEXT,
                driving_using_machine TEXT,
                how_to_use_medication TEXT,
                FOREIGN KEY (patient_id) REFERENCES patients(patient_id)
            )
        """)
        
        conn.commit()
        cursor.close()
        conn.close()

# Initialize session state variables
def init_session_state():
    if 'prescription_data' not in st.session_state:
        st.session_state.prescription_data = {
            "Nameoftablets": "", "ref": "", "Dose": "", "NumberofTablets": "",
            "Lot": "", "Issuedate": "", "ExpDate": "", "DailyDose": "",
            "sideEfect": "", "FurtherInformation": "", "StorageAdvice": "",
            "DrivingUsingMachine": "", "HowToUseMedication": "", 
            "PatientId": "", "nhsNumber": "", "PatientName": "",
            "DateOfBirth": "", "PatientAddress": "", "MedicalConditions": ""
        }

# Diabetes Prediction Function
def diabetes_prediction_tab():
    st.header("Diabetes Prediction")
    
    # Load or train the model
    try:
        model = pickle.load(open("rf_model_Diabetes", 'rb'))
        st.success("Diabetes prediction model loaded successfully!")
    except:
        st.warning("Model file not found. Training a new model...")
        model = train_diabetes_model()
    
    st.subheader("Enter Patient Details for Diabetes Prediction")
    
    col1, col2 = st.columns(2)
    
    with col1:
        gender = st.selectbox("Gender", ["Female", "Male", "Other"], key="diabetes_gender")
        age = st.slider("Age", 0, 100, 30, key="diabetes_age")
        hypertension = st.selectbox("Hypertension", ["No", "Yes"], key="diabetes_hypertension")
        heart_disease = st.selectbox("Heart Disease", ["No", "Yes"], key="diabetes_heart_disease")
    
    with col2:
        smoking_history = st.selectbox("Smoking History", 
                                      ["never", "current", "former", "ever", "not current", "No Info"], 
                                      key="diabetes_smoking")
        bmi = st.slider("BMI", 10.0, 50.0, 25.0, key="diabetes_bmi")
        hba1c_level = st.slider("HbA1c Level", 3.5, 9.0, 5.7, key="diabetes_hba1c")
        blood_glucose_level = st.slider("Blood Glucose Level", 80, 300, 120, key="diabetes_glucose")
    
    # Convert inputs to model format
    gender_encoded = 0 if gender == "Female" else 1 if gender == "Male" else 2
    hypertension_encoded = 1 if hypertension == "Yes" else 0
    heart_disease_encoded = 1 if heart_disease == "Yes" else 0
    
    # Encode smoking history
    smoking_mapping = {"never": 0, "current": 1, "former": 2, "ever": 3, "not current": 4, "No Info": 5}
    smoking_encoded = smoking_mapping[smoking_history]
    
    if st.button("Predict Diabetes Risk"):
        # Create input array
        input_data = np.array([[gender_encoded, age, hypertension_encoded, heart_disease_encoded, 
                               smoking_encoded, bmi, hba1c_level, blood_glucose_level]])
        
        # Make prediction
        prediction = model.predict(input_data)
        prediction_proba = model.predict_proba(input_data)
        
        # Display results
        st.subheader("Prediction Results")
        if prediction[0] == 1:
            st.error(f"High risk of diabetes: {prediction_proba[0][1]*100:.2f}% probability")
            st.warning("Recommendation: Please consult with a healthcare provider for further evaluation.")
        else:
            st.success(f"Low risk of diabetes: {prediction_proba[0][0]*100:.2f}% probability")
            st.info("Recommendation: Maintain a healthy lifestyle with regular exercise and balanced diet.")
        
        # Show feature importance
        st.subheader("Factors Influencing Prediction")
        feature_names = ["Gender", "Age", "Hypertension", "Heart Disease", 
                        "Smoking History", "BMI", "HbA1c Level", "Blood Glucose Level"]
        feature_importance = model.feature_importances_
        
        importance_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": feature_importance
        }).sort_values("Importance", ascending=False)
        
        st.bar_chart(importance_df.set_index("Feature"))

# Function to train diabetes model
def train_diabetes_model():
    # This function would train the model as in your provided code
    # For now, we'll just return a placeholder
    # In a real implementation, you would load your dataset and train the model
    st.info("Please ensure you have the 'diabetes_prediction_dataset.csv' file in the same directory.")
    return None

# Main function
def main():
    st.set_page_config(
        page_title="Hospital Management System",
        page_icon="🏥",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state
    init_session_state()
    
    # Create database if it doesn't exist
    create_database()
    
    # Create tables if they don't exist
    create_tables()
    
    # Sidebar with diseases and diabetes prediction
    with st.sidebar:
        st.header("Medical Conditions")
        
        # Create tabs in sidebar
        sidebar_tab1, sidebar_tab2 = st.tabs(["Conditions", "Diabetes Prediction"])
        
        with sidebar_tab1:
            st.subheader("Select Patient Conditions")
            
            # List of diseases
            diseases = [
                "Diabetes", "Lung Cancer", "Heart Attack", "Brain Tumor", 
                "Kidney Stones", "Hypertension", "Asthma", "Arthritis",
                "Stroke", "COVID-19", "HIV/AIDS", "Hepatitis",
                "Epilepsy", "Osteoporosis", "Alzheimer's", "Parkinson's",
                "Multiple Sclerosis", "Thyroid Disorders", "Anemia", "Migraine"
            ]
            
            selected_diseases = []
            cols = st.columns(2)
            for i, disease in enumerate(diseases):
                with cols[i % 2]:
                    if st.checkbox(disease, key=f"disease_{i}"):
                        selected_diseases.append(disease)
            
            # Store selected diseases in session state
            st.session_state.prescription_data['MedicalConditions'] = ", ".join(selected_diseases)
            
            if selected_diseases:
                st.write("**Selected Conditions:**")
                for disease in selected_diseases:
                    st.write(f"- {disease}")
        
        with sidebar_tab2:
            st.subheader("Diabetes Prediction")
            st.info("Use this tool to assess diabetes risk based on patient health metrics.")
            if st.button("Go to Diabetes Prediction"):
                st.session_state.current_tab = "Diabetes Prediction"
    
    # Header
    st.markdown(
        "<h1 style='text-align: center; color: red;'>HOSPITAL MANAGEMENT SYSTEM</h1>",
        unsafe_allow_html=True
    )
    
    # Check if we need to show diabetes prediction tab
    if hasattr(st.session_state, 'current_tab') and st.session_state.current_tab == "Diabetes Prediction":
        diabetes_prediction_tab()
        if st.button("Back to Main Dashboard"):
            st.session_state.current_tab = "Main"
            st.experimental_rerun()
    else:
        # Create tabs for different functionalities
        tab1, tab2, tab3, tab4 = st.tabs([
            "Patient Information", 
            "Prescription", 
            "View Data", 
            "Database Operations"
        ])
        
        # Patient Information Tab
        with tab1:
            st.header("Patient Information")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Personal Details")
                nhs_number = st.text_input("NHS Number", key="nhs_number")
                patient_name = st.text_input("Patient Name", key="patient_name")
                date_of_birth = st.date_input("Date of Birth", key="date_of_birth")
                patient_address = st.text_area("Patient Address", key="patient_address")
                
                # Display selected diseases
                if st.session_state.prescription_data['MedicalConditions']:
                    st.write("**Selected Medical Conditions:**")
                    st.write(st.session_state.prescription_data['MedicalConditions'])
                
                if st.button("Save Patient Information"):
                    if nhs_number and patient_name and date_of_birth and patient_address:
                        conn = connect_to_database()
                        if conn:
                            cursor = conn.cursor()
                            try:
                                cursor.execute(
                                    "INSERT INTO patients (nhs_number, patient_name, date_of_birth, patient_address, medical_conditions) VALUES (%s, %s, %s, %s, %s)",
                                    (nhs_number, patient_name, date_of_birth, patient_address, st.session_state.prescription_data['MedicalConditions'])
                                )
                                conn.commit()
                                patient_id = cursor.lastrowid
                                st.success(f"Patient information saved successfully! Patient ID: {patient_id}")
                                
                                # Store in session state
                                st.session_state.prescription_data['PatientId'] = patient_id
                                st.session_state.prescription_data['nhsNumber'] = nhs_number
                                st.session_state.prescription_data['PatientName'] = patient_name
                                st.session_state.prescription_data['DateOfBirth'] = str(date_of_birth)
                                st.session_state.prescription_data['PatientAddress'] = patient_address
                                
                            except mysql.connector.Error as e:
                                st.error(f"Error saving patient information: {e}")
                            finally:
                                cursor.close()
                                conn.close()
                    else:
                        st.error("Please fill all patient information fields")
            
            with col2:
                st.subheader("Search Patient")
                search_option = st.radio("Search by:", ("NHS Number", "Patient Name"))
                
                if search_option == "NHS Number":
                    search_nhs = st.text_input("Enter NHS Number")
                    if st.button("Search by NHS Number") and search_nhs:
                        conn = connect_to_database()
                        if conn:
                            cursor = conn.cursor(dictionary=True)
                            cursor.execute(
                                "SELECT * FROM patients WHERE nhs_number = %s", 
                                (search_nhs,)
                            )
                            patient = cursor.fetchone()
                            if patient:
                                st.write("**Patient Details:**")
                                st.write(f"**Patient ID:** {patient['patient_id']}")
                                st.write(f"**Name:** {patient['patient_name']}")
                                st.write(f"**Date of Birth:** {patient['date_of_birth']}")
                                st.write(f"**Address:** {patient['patient_address']}")
                                if patient['medical_conditions']:
                                    st.write(f"**Medical Conditions:** {patient['medical_conditions']}")
                                
                                # Store in session state
                                st.session_state.prescription_data['PatientId'] = patient['patient_id']
                                st.session_state.prescription_data['nhsNumber'] = patient['nhs_number']
                                st.session_state.prescription_data['PatientName'] = patient['patient_name']
                                st.session_state.prescription_data['DateOfBirth'] = str(patient['date_of_birth'])
                                st.session_state.prescription_data['PatientAddress'] = patient['patient_address']
                                st.session_state.prescription_data['MedicalConditions'] = patient['medical_conditions'] or ""
                            else:
                                st.warning("No patient found with this NHS number")
                            cursor.close()
                            conn.close()
                
                else:  # Search by Patient Name
                    search_name = st.text_input("Enter Patient Name")
                    if st.button("Search by Name") and search_name:
                        conn = connect_to_database()
                        if conn:
                            cursor = conn.cursor(dictionary=True)
                            cursor.execute(
                                "SELECT * FROM patients WHERE patient_name LIKE %s", 
                                (f"%{search_name}%",)
                            )
                            patients = cursor.fetchall()
                            if patients:
                                st.write("**Matching Patients:**")
                                for patient in patients:
                                    st.write(f"**ID:** {patient['patient_id']} | **Name:** {patient['patient_name']} | **NHS:** {patient['nhs_number']}")
                                    
                                    if st.button(f"Select {patient['patient_name']}", key=f"select_{patient['patient_id']}"):
                                        # Store in session state
                                        st.session_state.prescription_data['PatientId'] = patient['patient_id']
                                        st.session_state.prescription_data['nhsNumber'] = patient['nhs_number']
                                        st.session_state.prescription_data['PatientName'] = patient['patient_name']
                                        st.session_state.prescription_data['DateOfBirth'] = str(patient['date_of_birth'])
                                        st.session_state.prescription_data['PatientAddress'] = patient['patient_address']
                                        st.session_state.prescription_data['MedicalConditions'] = patient['medical_conditions'] or ""
                                        st.experimental_rerun()
                            else:
                                st.warning("No patients found with this name")
                            cursor.close()
                            conn.close()
        
        # Prescription Tab
        with tab2:
            st.header("Prescription Details")
            
            # Check if patient is selected
            if not st.session_state.prescription_data['PatientId']:
                st.warning("Please select or register a patient first in the Patient Information tab.")
            else:
                st.write(f"**Patient:** {st.session_state.prescription_data['PatientName']} (NHS: {st.session_state.prescription_data['nhsNumber']})")
                if st.session_state.prescription_data['MedicalConditions']:
                    st.write(f"**Medical Conditions:** {st.session_state.prescription_data['MedicalConditions']}")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    name_of_tablets = st.selectbox(
                        "Name of Tablets",
                        ["Nice", "Corona Vaccine", "Acetaminophen", "Adderall", "Amlodipine", "Ativan"],
                        key="name_of_tablets"
                    )
                    reference_no = st.text_input("Reference No.", key="reference_no")
                    dose = st.text_input("Dose", key="dose")
                    number_of_tablets = st.number_input("Number of Tablets", min_value=1, key="number_of_tablets")
                    lot = st.text_input("Lot", key="lot")
                    issue_date = st.date_input("Issue Date", key="issue_date")
                    exp_date = st.date_input("Expiry Date", key="exp_date")
                    daily_dose = st.text_input("Daily Dose", key="daily_dose")
                
                with col2:
                    side_effect = st.text_area("Side Effects", key="side_effect")
                    further_information = st.text_area("Further Information", key="further_information")
                    storage_advice = st.text_area("Storage Advice", key="storage_advice")
                    driving_using_machine = st.text_area("Driving Using Machine", key="driving_using_machine")
                    how_to_use_medication = st.text_area("How To Use Medication", key="how_to_use_medication")
                
                if st.button("Save Prescription"):
                    # Validate required fields
                    if not all([name_of_tablets, reference_no, dose, number_of_tablets]):
                        st.error("Please fill all required fields (Name of Tablets, Reference No, Dose, Number of Tablets)")
                    else:
                        conn = connect_to_database()
                        if conn:
                            cursor = conn.cursor()
                            try:
                                cursor.execute(
                                    """INSERT INTO prescriptions 
                                    (patient_id, name_of_tablets, reference_no, dose, number_of_tablets, 
                                    lot, issue_date, exp_date, daily_dose, side_effect, further_information, 
                                    storage_advice, driving_using_machine, how_to_use_medication) 
                                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                                    (
                                        st.session_state.prescription_data['PatientId'],
                                        name_of_tablets,
                                        reference_no,
                                        dose,
                                        number_of_tablets,
                                        lot,
                                        issue_date,
                                        exp_date,
                                        daily_dose,
                                        side_effect,
                                        further_information,
                                        storage_advice,
                                        driving_using_machine,
                                        how_to_use_medication
                                    )
                                )
                                conn.commit()
                                st.success("Prescription saved successfully!")
                                
                                # Generate prescription text
                                prescription_text = f"""
                                PRESCRIPTION DETAILS:
                                Patient: {st.session_state.prescription_data['PatientName']}
                                NHS Number: {st.session_state.prescription_data['nhsNumber']}
                                Date of Birth: {st.session_state.prescription_data['DateOfBirth']}
                                Address: {st.session_state.prescription_data['PatientAddress']}
                                Medical Conditions: {st.session_state.prescription_data['MedicalConditions']}
                                
                                Medication: {name_of_tablets}
                                Reference No: {reference_no}
                                Dose: {dose}
                                Number of Tablets: {number_of_tablets}
                                Lot: {lot}
                                Issue Date: {issue_date}
                                Expiry Date: {exp_date}
                                Daily Dose: {daily_dose}
                                
                                Side Effects: {side_effect}
                                Further Information: {further_information}
                                Storage Advice: {storage_advice}
                                Driving/Machine Usage: {driving_using_machine}
                                Usage Instructions: {how_to_use_medication}
                                """
                                
                                st.text_area("Prescription Summary", prescription_text, height=300)
                                
                            except mysql.connector.Error as e:
                                st.error(f"Error saving prescription: {e}")
                            finally:
                                cursor.close()
                                conn.close()
        
        # View Data Tab
        with tab3:
            st.header("View Patient and Prescription Data")
            
            view_option = st.radio("View:", ("Patients", "Prescriptions"))
            
            if view_option == "Patients":
                conn = connect_to_database()
                if conn:
                    cursor = conn.cursor(dictionary=True)
                    cursor.execute("SELECT * FROM patients")
                    patients = cursor.fetchall()
                    
                    if patients:
                        df = pd.DataFrame(patients)
                        st.dataframe(df)
                    else:
                        st.info("No patient records found.")
                    cursor.close()
                    conn.close()
            
            else:  # Prescriptions
                conn = connect_to_database()
                if conn:
                    cursor = conn.cursor(dictionary=True)
                    cursor.execute("""
                        SELECT p.prescription_id, pt.patient_name, p.name_of_tablets, p.reference_no, 
                               p.dose, p.number_of_tablets, p.issue_date, p.exp_date
                        FROM prescriptions p
                        JOIN patients pt ON p.patient_id = pt.patient_id
                    """)
                    prescriptions = cursor.fetchall()
                    
                    if prescriptions:
                        df = pd.DataFrame(prescriptions)
                        st.dataframe(df)
                        
                        # Option to view detailed prescription
                        prescription_ids = [str(p['prescription_id']) for p in prescriptions]
                        selected_id = st.selectbox("Select Prescription ID to view details", prescription_ids)
                        
                        if selected_id:
                            cursor.execute("""
                                SELECT p.*, pt.patient_name, pt.nhs_number, pt.date_of_birth, pt.patient_address, pt.medical_conditions
                                FROM prescriptions p
                                JOIN patients pt ON p.patient_id = pt.patient_id
                                WHERE p.prescription_id = %s
                            """, (selected_id,))
                            prescription = cursor.fetchone()
                            
                            if prescription:
                                st.subheader("Prescription Details")
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    st.write(f"**Patient:** {prescription['patient_name']}")
                                    st.write(f"**NHS Number:** {prescription['nhs_number']}")
                                    st.write(f"**Date of Birth:** {prescription['date_of_birth']}")
                                    st.write(f"**Address:** {prescription['patient_address']}")
                                    if prescription['medical_conditions']:
                                        st.write(f"**Medical Conditions:** {prescription['medical_conditions']}")
                                    st.write(f"**Medication:** {prescription['name_of_tablets']}")
                                    st.write(f"**Reference No:** {prescription['reference_no']}")
                                    st.write(f"**Dose:** {prescription['dose']}")
                                    st.write(f"**Number of Tablets:** {prescription['number_of_tablets']}")
                                
                                with col2:
                                    st.write(f"**Lot:** {prescription['lot']}")
                                    st.write(f"**Issue Date:** {prescription['issue_date']}")
                                    st.write(f"**Expiry Date:** {prescription['exp_date']}")
                                    st.write(f"**Daily Dose:** {prescription['daily_dose']}")
                                    st.write(f"**Side Effects:** {prescription['side_effect']}")
                                    st.write(f"**Further Information:** {prescription['further_information']}")
                                    st.write(f"**Storage Advice:** {prescription['storage_advice']}")
                                    st.write(f"**Driving/Machine Usage:** {prescription['driving_using_machine']}")
                                    st.write(f"**Usage Instructions:** {prescription['how_to_use_medication']}")
                    else:
                        st.info("No prescription records found.")
                    cursor.close()
                    conn.close()
        
        # Database Operations Tab
        with tab4:
            st.header("Database Operations")
            
            operation = st.selectbox("Select Operation", 
                                    ["Update Record", "Delete Record", "Clear All Data"])
            
            if operation == "Update Record":
                st.subheader("Update Patient Record")
                conn = connect_to_database()
                if conn:
                    cursor = conn.cursor(dictionary=True)
                    cursor.execute("SELECT patient_id, patient_name FROM patients")
                    patients = cursor.fetchall()
                    
                    if patients:
                        patient_options = {f"{p['patient_id']} - {p['patient_name']}": p['patient_id'] for p in patients}
                        selected_patient = st.selectbox("Select Patient", list(patient_options.keys()))
                        
                        if selected_patient:
                            patient_id = patient_options[selected_patient]
                            cursor.execute("SELECT * FROM patients WHERE patient_id = %s", (patient_id,))
                            patient = cursor.fetchone()
                            
                            if patient:
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    new_nhs = st.text_input("NHS Number", value=patient['nhs_number'])
                                    new_name = st.text_input("Patient Name", value=patient['patient_name'])
                                    new_dob = st.date_input("Date of Birth", value=patient['date_of_birth'])
                                
                                with col2:
                                    new_address = st.text_area("Patient Address", value=patient['patient_address'])
                                    new_conditions = st.text_area("Medical Conditions", value=patient['medical_conditions'] or "")
                                
                                if st.button("Update Patient Record"):
                                    cursor.execute(
                                        "UPDATE patients SET nhs_number = %s, patient_name = %s, date_of_birth = %s, patient_address = %s, medical_conditions = %s WHERE patient_id = %s",
                                        (new_nhs, new_name, new_dob, new_address, new_conditions, patient_id)
                                    )
                                    conn.commit()
                                    st.success("Patient record updated successfully!")
                            cursor.close()
                    else:
                        st.info("No patient records found.")
                    conn.close()
            
            elif operation == "Delete Record":
                st.subheader("Delete Record")
                delete_option = st.radio("Delete:", ("Patient", "Prescription"))
                
                conn = connect_to_database()
                if conn:
                    cursor = conn.cursor(dictionary=True)
                    
                    if delete_option == "Patient":
                        cursor.execute("SELECT patient_id, patient_name FROM patients")
                        patients = cursor.fetchall()
                        
                        if patients:
                            patient_options = {f"{p['patient_id']} - {p['patient_name']}": p['patient_id'] for p in patients}
                            selected_patient = st.selectbox("Select Patient to Delete", list(patient_options.keys()))
                            
                            if selected_patient and st.button("Delete Patient"):
                                patient_id = patient_options[selected_patient]
                                
                                # First delete related prescriptions
                                cursor.execute("DELETE FROM prescriptions WHERE patient_id = %s", (patient_id,))
                                # Then delete the patient
                                cursor.execute("DELETE FROM patients WHERE patient_id = %s", (patient_id,))
                                conn.commit()
                                st.success("Patient and related prescriptions deleted successfully!")
                        else:
                            st.info("No patient records found.")
                    
                    else:  # Delete Prescription
                        cursor.execute("""
                            SELECT p.prescription_id, pt.patient_name, p.name_of_tablets 
                            FROM prescriptions p
                            JOIN patients pt ON p.patient_id = pt.patient_id
                        """)
                        prescriptions = cursor.fetchall()
                        
                        if prescriptions:
                            prescription_options = {f"{p['prescription_id']} - {p['patient_name']} - {p['name_of_tablets']}": p['prescription_id'] for p in prescriptions}
                            selected_prescription = st.selectbox("Select Prescription to Delete", list(prescription_options.keys()))
                            
                            if selected_prescription and st.button("Delete Prescription"):
                                prescription_id = prescription_options[selected_prescription]
                                cursor.execute("DELETE FROM prescriptions WHERE prescription_id = %s", (prescription_id,))
                                conn.commit()
                                st.success("Prescription deleted successfully!")
                        else:
                            st.info("No prescription records found.")
                    
                    cursor.close()
                    conn.close()
            
            else:  # Clear All Data
                st.subheader("Clear All Data")
                st.warning("This action will delete ALL data from the database. This cannot be undone!")
                
                if st.checkbox("I understand this action cannot be undone"):
                    if st.button("Clear All Data"):
                        conn = connect_to_database()
                        if conn:
                            cursor = conn.cursor()
                            try:
                                # Delete all prescriptions first (due to foreign key constraint)
                                cursor.execute("DELETE FROM prescriptions")
                                # Then delete all patients
                                cursor.execute("DELETE FROM patients")
                                conn.commit()
                                st.success("All data has been cleared from the database.")
                            except mysql.connector.Error as e:
                                st.error(f"Error clearing data: {e}")
                            finally:
                                cursor.close()
                                conn.close()

if __name__ == "__main__":
    main()