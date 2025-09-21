import streamlit as st
import mysql.connector
from datetime import datetime
import pandas as pd
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

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
                patient_address TEXT
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
        
        # Create diabetes predictions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS diabetes_predictions (
                prediction_id INT AUTO_INCREMENT PRIMARY KEY,
                patient_id INT,
                gender VARCHAR(10),
                age INT,
                hypertension BOOLEAN,
                heart_disease BOOLEAN,
                smoking_history VARCHAR(50),
                bmi FLOAT,
                hba1c_level FLOAT,
                blood_glucose_level INT,
                prediction_result BOOLEAN,
                prediction_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
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
            "DateOfBirth": "", "PatientAddress": ""
        }
    
    # Initialize diabetes prediction model
    if 'diabetes_model' not in st.session_state:
        try:
            # Load your trained model here
            # For demonstration, I'm creating a placeholder model
            # Replace this with your actual model loading code
            st.session_state.diabetes_model = LogisticRegression()
            st.session_state.diabetes_model_loaded = True
        except:
            st.session_state.diabetes_model_loaded = False
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

# Diabetes prediction function
def predict_diabetes(gender, age, hypertension, heart_disease, smoking_history, bmi, hba1c_level, blood_glucose_level):
    try:
        # Encode categorical variables (same as in your training)
        gender_encoded = 1 if gender == "Male" else (0 if gender == "Female" else 2)
        
        smoking_mapping = {
            "never": 0,
            "former": 1,
            "current": 2,
            "not current": 3,
            "ever": 4
        }
        smoking_encoded = smoking_mapping.get(smoking_history.lower(), 0)
        
        # Create feature array
        features = np.array([[gender_encoded, age, hypertension, heart_disease, 
                             smoking_encoded, bmi, hba1c_level, blood_glucose_level]])
        
        # Make prediction
        prediction = st.session_state.diabetes_model.predict(features)
        probability = st.session_state.diabetes_model.predict_proba(features)
        
        return prediction[0], probability[0][1]
    except Exception as e:
        st.error(f"Error in prediction: {e}")
        return None, None

# Diabetes chatbot function
def diabetes_chatbot():
    st.header("Diabetes Information Chatbot")
    
    # Initialize chat history
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = [
            {"role": "assistant", "content": "Hello! I'm here to provide information about diabetes. How can I help you today?"}
        ]
    
    # Display chat messages from history on app rerun
    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # React to user input
    if prompt := st.chat_input("Ask about diabetes..."):
        # Display user message in chat message container
        st.chat_message("user").markdown(prompt)
        # Add user message to chat history
        st.session_state.chat_messages.append({"role": "user", "content": prompt})
        
        # Simple response logic - you can expand this with more sophisticated responses
        response = ""
        prompt_lower = prompt.lower()
        
        if "symptom" in prompt_lower:
            response = "Common symptoms of diabetes include frequent urination, excessive thirst, extreme hunger, unexplained weight loss, fatigue, blurred vision, and slow-healing sores."
        elif "type 1" in prompt_lower:
            response = "Type 1 diabetes is an autoimmune condition where the pancreas produces little or no insulin. It's usually diagnosed in children and young adults."
        elif "type 2" in prompt_lower:
            response = "Type 2 diabetes is a chronic condition that affects the way your body processes blood sugar (glucose). It's characterized by insulin resistance."
        elif "prevent" in prompt_lower or "avoid" in prompt_lower:
            response = "You can reduce your risk of type 2 diabetes by maintaining a healthy weight, eating a balanced diet, exercising regularly, and avoiding smoking."
        elif "diet" in prompt_lower or "food" in prompt_lower:
            response = "A diabetes-friendly diet includes whole grains, fruits, vegetables, lean proteins, and healthy fats. Limit sugary foods and refined carbohydrates."
        elif "treatment" in prompt_lower or "medication" in prompt_lower:
            response = "Diabetes treatment may include lifestyle changes, oral medications, insulin therapy, or other injectable medications, depending on the type and severity."
        elif "blood sugar" in prompt_lower or "glucose" in prompt_lower:
            response = "Normal fasting blood sugar is typically between 70-100 mg/dL. For people with diabetes, target ranges are usually 80-130 mg/dL before meals and below 180 mg/dL after meals."
        elif "insulin" in prompt_lower:
            response = "Insulin is a hormone that helps glucose enter your cells to be used for energy. People with type 1 diabetes need insulin therapy, and some with type 2 diabetes may also require it."
        else:
            response = "I'm sorry, I don't have specific information about that. Could you ask about diabetes symptoms, types, prevention, diet, treatment, or blood sugar management?"
        
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st.markdown(response)
        # Add assistant response to chat history
        st.session_state.chat_messages.append({"role": "assistant", "content": response})






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
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.selectbox("Choose a module", 
                                   ["Hospital Management", "Diabetes Prediction", "Diabetes Chatbot"])
    
    # Header
    st.markdown(
        "<h1 style='text-align: center; color: red;'>HOSPITAL MANAGEMENT SYSTEM</h1>",
        unsafe_allow_html=True
    )
    
    # Hospital Management Module
    if app_mode == "Hospital Management":
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
                
                if st.button("Save Patient Information"):
                    if nhs_number and patient_name and date_of_birth and patient_address:
                        conn = connect_to_database()
                        if conn:
                            cursor = conn.cursor()
                            try:
                                cursor.execute(
                                    "INSERT INTO patients (nhs_number, patient_name, date_of_birth, patient_address) VALUES (%s, %s, %s, %s)",
                                    (nhs_number, patient_name, date_of_birth, patient_address)
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
                                
                                # Store in session state
                                st.session_state.prescription_data['PatientId'] = patient['patient_id']
                                st.session_state.prescription_data['nhsNumber'] = patient['nhs_number']
                                st.session_state.prescription_data['PatientName'] = patient['patient_name']
                                st.session_state.prescription_data['DateOfBirth'] = str(patient['date_of_birth'])
                                st.session_state.prescription_data['PatientAddress'] = patient['patient_address']
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
            
            view_option = st.radio("View:", ("Patients", "Prescriptions", "Diabetes Predictions"))
            
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
            
            elif view_option == "Prescriptions":
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
                                SELECT p.*, pt.patient_name, pt.nhs_number, pt.date_of_birth, pt.patient_address
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
            
            else:  # Diabetes Predictions
                conn = connect_to_database()
                if conn:
                    cursor = conn.cursor(dictionary=True)
                    cursor.execute("""
                        SELECT d.*, p.patient_name, p.nhs_number
                        FROM diabetes_predictions d
                        JOIN patients p ON d.patient_id = p.patient_id
                        ORDER BY d.prediction_date DESC
                    """)
                    predictions = cursor.fetchall()
                    
                    if predictions:
                        df = pd.DataFrame(predictions)
                        st.dataframe(df)
                    else:
                        st.info("No diabetes prediction records found.")
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
                                
                                if st.button("Update Patient Record"):
                                    cursor.execute(
                                        "UPDATE patients SET nhs_number = %s, patient_name = %s, date_of_birth = %s, patient_address = %s WHERE patient_id = %s",
                                        (new_nhs, new_name, new_dob, new_address, patient_id)
                                    )
                                    conn.commit()
                                    st.success("Patient record updated successfully!")
                            cursor.close()
                    else:
                        st.info("No patient records found.")
                    conn.close()
            
            elif operation == "Delete Record":
                st.subheader("Delete Record")
                delete_option = st.radio("Delete:", ("Patient", "Prescription", "Diabetes Prediction"))
                
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
                                
                                # First delete related records
                                cursor.execute("DELETE FROM prescriptions WHERE patient_id = %s", (patient_id,))
                                cursor.execute("DELETE FROM diabetes_predictions WHERE patient_id = %s", (patient_id,))
                                # Then delete the patient
                                cursor.execute("DELETE FROM patients WHERE patient_id = %s", (patient_id,))
                                conn.commit()
                                st.success("Patient and related records deleted successfully!")
                        else:
                            st.info("No patient records found.")
                    
                    elif delete_option == "Prescription":
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
                    
                    else:  # Delete Diabetes Prediction
                        cursor.execute("""
                            SELECT d.prediction_id, p.patient_name, d.prediction_date, d.prediction_result
                            FROM diabetes_predictions d
                            JOIN patients p ON d.patient_id = p.patient_id
                        """)
                        predictions = cursor.fetchall()
                        
                        if predictions:
                            prediction_options = {f"{p['prediction_id']} - {p['patient_name']} - {p['prediction_date']}": p['prediction_id'] for p in predictions}
                            selected_prediction = st.selectbox("Select Prediction to Delete", list(prediction_options.keys()))
                            
                            if selected_prediction and st.button("Delete Prediction"):
                                prediction_id = prediction_options[selected_prediction]
                                cursor.execute("DELETE FROM diabetes_predictions WHERE prediction_id = %s", (prediction_id,))
                                conn.commit()
                                st.success("Diabetes prediction record deleted successfully!")
                        else:
                            st.info("No diabetes prediction records found.")
                    
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
                                # Delete all records in the correct order (due to foreign key constraints)
                                cursor.execute("DELETE FROM diabetes_predictions")
                                cursor.execute("DELETE FROM prescriptions")
                                cursor.execute("DELETE FROM patients")
                                conn.commit()
                                st.success("All data has been cleared from the database.")
                            except mysql.connector.Error as e:
                                st.error(f"Error clearing data: {e}")
                            finally:
                                cursor.close()
                                conn.close()
    
    # Diabetes Prediction Module
    elif app_mode == "Diabetes Prediction":
        st.header("Diabetes Prediction")
        
        # Check if model is loaded
        if not st.session_state.diabetes_model_loaded:
            st.error("Diabetes prediction model could not be loaded. Please check the model file.")
        else:
            # Patient selection
            conn = connect_to_database()
            if conn:
                cursor = conn.cursor(dictionary=True)
                cursor.execute("SELECT patient_id, patient_name FROM patients")
                patients = cursor.fetchall()
                
                if patients:
                    patient_options = {f"{p['patient_id']} - {p['patient_name']}": p['patient_id'] for p in patients}
                    patient_options["New Patient"] = "new"
                    selected_patient = st.selectbox("Select Patient or 'New Patient'", list(patient_options.keys()))
                    
                    if selected_patient:
                        if patient_options[selected_patient] == "new":
                            st.info("Please register the patient in the Hospital Management module first.")
                        else:
                            patient_id = patient_options[selected_patient]
                            cursor.execute("SELECT * FROM patients WHERE patient_id = %s", (patient_id,))
                            patient = cursor.fetchone()
                            
                            if patient:
                                st.write(f"**Patient:** {patient['patient_name']}")
                                st.write(f"**NHS Number:** {patient['nhs_number']}")
                                st.write(f"**Date of Birth:** {patient['date_of_birth']}")
                
                cursor.close()
                conn.close()
            
            # Prediction form
            with st.form("diabetes_prediction_form"):
                st.subheader("Patient Health Information")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    gender = st.selectbox("Gender", ["Female", "Male", "Other"])
                    age = st.slider("Age", 0, 100, 30)
                    hypertension = st.radio("Hypertension", ["No", "Yes"])
                    heart_disease = st.radio("Heart Disease", ["No", "Yes"])
                    smoking_history = st.selectbox("Smoking History", 
                                                 ["never", "former", "current", "not current", "ever"])
                
                with col2:
                    bmi = st.slider("BMI", 10.0, 50.0, 25.0)
                    hba1c_level = st.slider("HbA1c Level", 3.0, 15.0, 5.7)
                    blood_glucose_level = st.slider("Blood Glucose Level", 80, 300, 120)
                
                submitted = st.form_submit_button("Predict Diabetes Risk")
                
                if submitted:
                    # Convert yes/no to boolean
                    hypertension_bool = 1 if hypertension == "Yes" else 0
                    heart_disease_bool = 1 if heart_disease == "Yes" else 0
                    
                    # Make prediction
                    prediction, probability = predict_diabetes(
                        gender, age, hypertension_bool, heart_disease_bool, 
                        smoking_history, bmi, hba1c_level, blood_glucose_level
                    )
                    
                    if prediction is not None:
                        # Display results
                        if prediction == 1:
                            st.error(f"High risk of diabetes. Probability: {probability:.2%}")
                        else:
                            st.success(f"Low risk of diabetes. Probability: {probability:.2%}")
                        
                        # Save prediction to database if patient is selected
                        if selected_patient and patient_options[selected_patient] != "new":
                            conn = connect_to_database()
                            if conn:
                                cursor = conn.cursor()
                                try:
                                    cursor.execute(
                                        """INSERT INTO diabetes_predictions 
                                        (patient_id, gender, age, hypertension, heart_disease, 
                                        smoking_history, bmi, hba1c_level, blood_glucose_level, prediction_result) 
                                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                                        (
                                            patient_id,
                                            gender,
                                            age,
                                            hypertension_bool,
                                            heart_disease_bool,
                                            smoking_history,
                                            bmi,
                                            hba1c_level,
                                            blood_glucose_level,
                                            prediction
                                        )
                                    )
                                    conn.commit()
                                    st.success("Prediction saved to database!")
                                except mysql.connector.Error as e:
                                    st.error(f"Error saving prediction: {e}")
                                finally:
                                    cursor.close()
                                    conn.close()
    
    # Diabetes Chatbot Module
    elif app_mode == "Diabetes Chatbot":
        diabetes_chatbot()

if __name__ == "__main__":
    main()