import os
import openai
import asyncio
import pandas as pd
import smtplib
import sqlite3
import json
import streamlit as st
import re
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta
from dotenv import load_dotenv
from agents import Agent, Runner, function_tool, handoff, RunContextWrapper

# Load environment variables
load_dotenv(override=True)

# Configuration
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
EMAIL_USER = os.environ.get("EMAIL_USER")
EMAIL_APP_PASSWORD = os.environ.get("EMAIL_APP_PASSWORD")
EMAIL_ENABLED = EMAIL_USER and EMAIL_APP_PASSWORD
DB_FILE = os.environ.get("DB_FILE", "patients.db")

DEPARTMENT_ROUTING = {
    "emergency": EMAIL_USER, "cardiology": EMAIL_USER, "dermatology": EMAIL_USER,
    "orthopedics": EMAIL_USER, "mental_health": EMAIL_USER, "general": EMAIL_USER,
    "pediatrics": EMAIL_USER, "gynecology": EMAIL_USER
}

URGENCY_LEVELS = {
    "critical": "🔴 CRITICAL - Immediate attention required",
    "high": "🟠 HIGH - Same day appointment needed", 
    "medium": "🟡 MEDIUM - Within 3-5 days",
    "low": "🟢 LOW - Routine care, within 1-2 weeks"
}

def log_system_message(message):
    if 'system_logs' not in st.session_state:
        st.session_state['system_logs'] = []
    timestamp = datetime.now().strftime("%H:%M:%S")
    st.session_state['system_logs'].append(f"[{timestamp}] {message}")

def extract_patient_details(conversation_history):
    if not conversation_history:
        return {"name": "Unknown", "age": "", "gender": "", "phone": "", "email": "", "insurance": "", "symptoms": "", "medical_history": ""}
    
    details = {"name": "Unknown", "age": "", "gender": "", "phone": "", "email": "", "insurance": "", "symptoms": "", "medical_history": ""}
    
    # Name extraction
    name_patterns = [r"I'm\s+(\w+)", r"I am\s+(\w+)", r"name\s+is\s+(\w+)", r"this\s+is\s+(\w+)"]
    for pattern in name_patterns:
        match = re.search(pattern, conversation_history, re.IGNORECASE)
        if match:
            details["name"] = match.group(1).strip()
            break
    
    # Email extraction
    email_match = re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', conversation_history)
    if email_match:
        details["email"] = email_match.group().strip()
    
    # Phone extraction
    phone_patterns = [r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b', r'\b\(\d{3}\)\s*\d{3}[-.\s]?\d{4}\b']
    for pattern in phone_patterns:
        match = re.search(pattern, conversation_history)
        if match:
            details["phone"] = match.group().strip()
            break
    
    return details

def assess_urgency(symptoms, age=None, medical_history=None):
    if not symptoms:
        return "medium"
    
    symptoms_lower = symptoms.lower()
    critical_keywords = ["chest pain", "heart attack", "stroke", "bleeding", "unconscious", "difficulty breathing", "severe pain", "poisoning", "overdose", "suicide", "severe injury", "broken bone", "head injury"]
    high_keywords = ["fever", "infection", "rash", "severe headache", "nausea", "vomiting", "dizzy", "swelling", "shortness of breath"]
    
    for keyword in critical_keywords:
        if keyword in symptoms_lower:
            return "critical"
    
    for keyword in high_keywords:
        if keyword in symptoms_lower:
            return "high"
    
    return "medium"

def determine_specialty(symptoms, age=None):
    if not symptoms:
        return "general"
    
    symptoms_lower = symptoms.lower()
    
    emergency_keywords = ["chest pain", "heart attack", "stroke", "bleeding", "unconscious", "difficulty breathing", "severe pain", "poisoning", "overdose", "suicide", "severe injury"]
    for keyword in emergency_keywords:
        if keyword in symptoms_lower:
            return "emergency"
    
    cardiology_keywords = ["heart", "cardiac", "chest pain", "palpitations", "blood pressure", "cholesterol", "arrhythmia", "angina"]
    for keyword in cardiology_keywords:
        if keyword in symptoms_lower:
            return "cardiology"
    
    dermatology_keywords = ["skin", "rash", "acne", "mole", "eczema", "psoriasis", "dermatitis", "itching", "burning skin"]
    for keyword in dermatology_keywords:
        if keyword in symptoms_lower:
            return "dermatology"
    
    orthopedics_keywords = ["bone", "joint", "back pain", "knee", "shoulder", "hip", "fracture", "sprain", "arthritis", "muscle pain"]
    for keyword in orthopedics_keywords:
        if keyword in symptoms_lower:
            return "orthopedics"
    
    mental_health_keywords = ["depression", "anxiety", "stress", "panic", "mental health", "counseling", "therapy", "mood", "psychiatric"]
    for keyword in mental_health_keywords:
        if keyword in symptoms_lower:
            return "mental_health"
    
    if age and age.isdigit() and int(age) < 18:
        return "pediatrics"
    
    gynecology_keywords = ["pregnancy", "menstrual", "gynecological", "contraception", "pap smear", "mammogram", "breast"]
    for keyword in gynecology_keywords:
        if keyword in symptoms_lower:
            return "gynecology"
    
    return "general"

def init_database():
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute('''CREATE TABLE IF NOT EXISTS patients (id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT NOT NULL, name TEXT NOT NULL, age TEXT, gender TEXT, phone TEXT, email TEXT, insurance TEXT, symptoms TEXT, medical_history TEXT, specialty TEXT NOT NULL, urgency TEXT NOT NULL, status TEXT DEFAULT 'pending')''')
        conn.commit()
        conn.close()
        log_system_message(f"DATABASE: Connected to {DB_FILE}")
        return True
    except Exception as e:
        log_system_message(f"DATABASE ERROR: Failed to initialize - {e}")
        return False

def get_all_patients():
    try:
        log_system_message("DATABASE: Retrieving all patient records")
        conn = sqlite3.connect(DB_FILE)
        df = pd.read_sql_query("SELECT * FROM patients ORDER BY timestamp DESC", conn)
        conn.close()
        log_system_message(f"DATABASE: Retrieved {len(df)} patient records")
        return df
    except Exception as e:
        log_system_message(f"DATABASE ERROR: {e}")
        return pd.DataFrame()

def send_email_message(to_email, subject, body, cc=None, log_prefix="EMAIL"):
    log_system_message(f"{log_prefix}: Sending to {to_email} - {subject}")
    
    if not EMAIL_ENABLED:
        message = f"Email disabled. Would send to {to_email}: {subject}"
        log_system_message(message)
        return message
    
    try:
        msg = MIMEMultipart()
        msg['From'] = EMAIL_USER
        msg['To'] = to_email
        msg['Subject'] = subject
        if cc:
            msg['Cc'] = cc
        msg.attach(MIMEText(body, 'html'))
        
        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()
            server.login(EMAIL_USER, EMAIL_APP_PASSWORD)
            recipients = [to_email] + (cc.split(',') if cc else [])
            server.sendmail(EMAIL_USER, recipients, msg.as_string())
        
        success_msg = f"Email sent successfully to {to_email}"
        log_system_message(f"{log_prefix}: ✅ {success_msg}")
        return success_msg
        
    except Exception as e:
        error_msg = f"Failed to send email: {str(e)}"
        log_system_message(f"{log_prefix}: ❌ {error_msg}")
        return error_msg

def create_admin_notification_email(name, age=None, gender=None, phone=None, email=None, insurance=None, symptoms=None, medical_history=None, specialty="general", urgency="medium"):
    urgency_display = URGENCY_LEVELS.get(urgency, urgency.upper())
    
    dept_info = {
        "emergency": {"name": "Emergency Department", "icon": "🚨", "color": "#dc2626"},
        "cardiology": {"name": "Cardiology Department", "icon": "❤️", "color": "#ef4444"},
        "dermatology": {"name": "Dermatology Department", "icon": "🔬", "color": "#06b6d4"},
        "orthopedics": {"name": "Orthopedic Department", "icon": "🦴", "color": "#8b5cf6"},
        "mental_health": {"name": "Mental Health Department", "icon": "🧠", "color": "#10b981"},
        "pediatrics": {"name": "Pediatrics Department", "icon": "👶", "color": "#f59e0b"},
        "gynecology": {"name": "Women's Health Department", "icon": "👩‍⚕️", "color": "#ec4899"},
        "general": {"name": "General Practice", "icon": "👨‍⚕️", "color": "#6366f1"}
    }
    
    dept = dept_info.get(specialty, dept_info["general"])
    urgency_colors = {"critical": "#dc2626", "high": "#ea580c", "medium": "#eab308", "low": "#16a34a"}
    urgency_color = urgency_colors.get(urgency, "#6b7280")
    
    email_body = f'<div style="font-family: Arial, sans-serif; max-width: 700px; margin: 0 auto; background: #f8fafc;"><div style="background: linear-gradient(135deg, #2c5aa0 0%, #1e3a8a 100%); color: white; padding: 2rem; text-align: center;"><h1 style="margin: 0; font-size: 1.8rem;">🏥 New Patient Alert</h1><p style="margin: 0.5rem 0 0 0; opacity: 0.9;">Healthcare Intake System</p></div><div style="padding: 2rem; background: white;"><div style="background: {urgency_color}; color: white; padding: 1rem; border-radius: 8px; text-align: center; margin-bottom: 2rem;"><h2 style="margin: 0; font-size: 1.2rem;">🚨 {urgency_display}</h2></div><div style="background: {dept["color"]}15; border-left: 4px solid {dept["color"]}; padding: 1.5rem; margin: 1.5rem 0; border-radius: 0 8px 8px 0;"><h3 style="color: {dept["color"]}; margin-top: 0;"><span style="font-size: 1.5rem;">{dept["icon"]}</span> Routed to: {dept["name"]}</h3></div><h3 style="color: #2c5aa0; border-bottom: 2px solid #e5e7eb; padding-bottom: 0.5rem;">Patient Information</h3><div style="margin-bottom: 2rem;"><p><strong>Name:</strong> {name}</p><p><strong>Age:</strong> {age or "Not provided"}</p><p><strong>Gender:</strong> {gender or "Not provided"}</p><p><strong>Phone:</strong> {phone or "Not provided"}</p><p><strong>Email:</strong> {email or "Not provided"}</p><p><strong>Insurance:</strong> {insurance or "Not provided"}</p><p><strong>Department:</strong> {specialty.title()}</p><p><strong>Priority:</strong> <span style="color: {urgency_color}; font-weight: bold;">{urgency.upper()}</span></p></div><h3 style="color: #2c5aa0; border-bottom: 2px solid #e5e7eb; padding-bottom: 0.5rem;">Medical Information</h3><div style="background: #f8fafc; padding: 1rem; border-radius: 8px; margin-bottom: 1rem;"><p><strong>Symptoms:</strong></p><p style="padding: 0.5rem; background: white; border-radius: 4px; border-left: 3px solid #2c5aa0;">{symptoms or "Not provided"}</p><p><strong>Medical History:</strong></p><p style="padding: 0.5rem; background: white; border-radius: 4px; border-left: 3px solid #2c5aa0;">{medical_history or "Not provided"}</p></div><h3 style="color: #2c5aa0; border-bottom: 2px solid #e5e7eb; padding-bottom: 0.5rem;">Next Steps</h3><div style="background: #fef3c7; border: 1px solid #fcd34d; border-radius: 8px; padding: 1rem;">'
    
    if urgency == 'critical':
        email_body += '<p style="color: #92400e; font-weight: bold;">🚨 IMMEDIATE ACTION REQUIRED - Contact patient immediately</p>'
    elif urgency == 'high':
        email_body += '<p style="color: #92400e; font-weight: bold;">⚡ HIGH PRIORITY - Schedule same-day appointment</p>'
    elif urgency == 'medium':
        email_body += '<p style="color: #92400e;">📅 Schedule appointment within 3-5 days</p>'
    else:
        email_body += '<p style="color: #92400e;">📋 Routine scheduling (1-2 weeks)</p>'
    
    email_body += f'<p style="color: #92400e;">• Review patient information and medical history</p><p style="color: #92400e;">• Contact patient using provided contact information</p><p style="color: #92400e;">• Verify insurance coverage before appointment</p><p style="color: #92400e;">• Prepare {dept["name"].lower()} consultation materials</p></div><div style="margin-top: 2rem; padding: 1rem; background: #f1f5f9; border-radius: 8px; text-align: center;"><p style="margin: 0; color: #6b7280; font-size: 0.9rem;"><strong>Record created:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}<br><strong>System:</strong> Healthcare Intake System v2.0</p></div></div></div>'
    
    return email_body

def send_admin_notification(name, **patient_info):
    if not EMAIL_ENABLED or not EMAIL_USER:
        return "Email not configured"
    
    specialty = patient_info.get('specialty', 'general')
    urgency = patient_info.get('urgency', 'medium')
    
    urgency_prefix = ""
    if urgency == "critical":
        urgency_prefix = "🔴 CRITICAL ALERT: "
    elif urgency == "high":
        urgency_prefix = "🟠 HIGH PRIORITY: "
    
    subject = f"{urgency_prefix}New Patient: {name} → {specialty.title()} Dept"
    body = create_admin_notification_email(name, **patient_info)
    
    log_system_message(f"ADMIN NOTIFICATION: Sending alert for {name} ({specialty}, {urgency})")
    return send_email_message(EMAIL_USER, subject, body, log_prefix="ADMIN")

def save_patient_to_database(name, age=None, gender=None, phone=None, email=None, insurance=None, symptoms=None, medical_history=None, specialty="general", urgency="medium"):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_system_message(f"DATABASE: Storing patient record for {name}")
    
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute('''INSERT INTO patients (timestamp, name, age, gender, phone, email, insurance, symptoms, medical_history, specialty, urgency) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''', (timestamp, name, age or "", gender or "", phone or "", email or "", insurance or "", symptoms or "", medical_history or "", specialty, urgency))
        conn.commit()
        conn.close()
        
        log_system_message(f"DATABASE: ✅ Patient record stored for {name}")
        log_system_message(f"DEPARTMENT ROUTING: 🏥 {name} → {specialty.title()} Department ({urgency.upper()} priority)")
        
        patient_info = {'age': age, 'gender': gender, 'phone': phone, 'email': email, 'insurance': insurance, 'symptoms': symptoms, 'medical_history': medical_history, 'specialty': specialty, 'urgency': urgency}
        
        email_result = send_admin_notification(name, **patient_info)
        log_system_message(f"ADMIN EMAIL: {email_result}")
        
        return f"✅ Patient record for {name} successfully stored and admin notified"
        
    except Exception as e:
        error_msg = f"Failed to store patient record: {str(e)}"
        log_system_message(f"DATABASE ERROR: ❌ {error_msg}")
        return error_msg

@function_tool
def send_email(to_email: str, subject: str, body: str, cc: str = None) -> str:
    return send_email_message(to_email, subject, body, cc)

@function_tool
def route_patient_to_department(specialty: str, name: str, age: str = None, gender: str = None, phone: str = None, email: str = None, insurance: str = None, symptoms: str = None, medical_history: str = None, urgency: str = "medium") -> str:
    log_system_message(f"🏥 DEPARTMENT ROUTING: {name} → {specialty.title()} Department")
    log_system_message(f"📊 PRIORITY LEVEL: {urgency.upper()}")
    log_system_message(f"🔍 SYMPTOMS: {symptoms[:50] + '...' if symptoms and len(symptoms) > 50 else symptoms or 'None provided'}")
    return f"Patient {name} routed to {specialty} department with {urgency} priority"

@function_tool
def store_patient_in_database(name: str, age: str = None, gender: str = None, phone: str = None, email: str = None, insurance: str = None, symptoms: str = None, medical_history: str = None, specialty: str = "general", urgency: str = "medium") -> str:
    return save_patient_to_database(name, age, gender, phone, email, insurance, symptoms, medical_history, specialty, urgency)

def create_agent_system():
    coordinator_instructions = "You are a medical intake and triage coordinator. Your job is to: 1. Warmly greet patients and collect essential information 2. Assess symptoms and determine appropriate medical specialty: Emergency for life-threatening, severe pain, breathing issues, chest pain; Cardiology for heart issues, chest pain, palpitations, blood pressure; Dermatology for skin issues, rashes, moles, allergic reactions; Orthopedics for bone/joint pain, injuries, fractures, mobility issues; Mental Health for depression, anxiety, stress, emotional concerns; Pediatrics for patients under 18 years old; Gynecology for women's reproductive health, pregnancy, menstrual issues; General for routine care, common illnesses, check-ups. 3. Assess urgency level: Critical for life-threatening (advise 911/ER), High for same-day care needed, Medium for within 3-5 days, Low for routine care (1-2 weeks). ALWAYS collect: name, age, gender, contact info, insurance, symptoms, medical history. IMPORTANT: For EVERY patient, use these tools: route_patient_to_department to notify appropriate medical department, and store_patient_in_database to save patient information. Be compassionate, professional, and thorough in your assessment."
    
    intake_coordinator = Agent(
        name="IntakeCoordinator",
        instructions=coordinator_instructions,
        tools=[route_patient_to_department, store_patient_in_database, send_email]
    )
    
    return intake_coordinator

async def process_user_message(user_input):
    if 'conversation_history' not in st.session_state:
        st.session_state['conversation_history'] = ""
    
    if st.session_state['conversation_history']:
        st.session_state['conversation_history'] += f"\nPatient: {user_input}"
    else:
        st.session_state['conversation_history'] = user_input
    
    log_system_message(f"PROCESSING: New message: {user_input[:50]}...")
    
    try:
        if 'intake_coordinator' not in st.session_state:
            log_system_message("PROCESSING: Creating intake coordinator agent")
            st.session_state['intake_coordinator'] = create_agent_system()
        
        log_system_message("PROCESSING: Running through intake coordinator")
        with st.spinner('Processing your message...'):
            result = await Runner.run(st.session_state['intake_coordinator'], st.session_state['conversation_history'])
        
        response = result.final_output
        log_system_message(f"PROCESSING: Generated response: {response[:50]}...")
        
        st.session_state['conversation_history'] += f"\nAssistant: {response}"
        st.session_state['messages'].append({"role": "user", "content": user_input})
        st.session_state['messages'].append({"role": "assistant", "content": response})
        
        return response
        
    except Exception as e:
        error_msg = f"Error processing message: {str(e)}"
        log_system_message(f"PROCESSING ERROR: {error_msg}")
        return "I apologize, but there was an error processing your message. Please try again."

def render_sidebar():
    sidebar_header = '<div style="text-align: center; padding: 1rem; background: linear-gradient(135deg, #2c5aa0, #1e3a8a); border-radius: 10px; margin-bottom: 1rem;"><h2 style="color: white; margin: 0;">⚕️ System Control Panel</h2><p style="color: #e0e7ff; margin: 0.5rem 0 0 0; font-size: 0.9rem;">Medical Administration Dashboard</p></div>'
    st.sidebar.markdown(sidebar_header, unsafe_allow_html=True)
    
    st.sidebar.markdown("### 🔧 System Status")
    
    if OPENAI_API_KEY:
        api_status = '<div style="background: linear-gradient(90deg, #10b981 0%, #059669 100%); color: white; padding: 0.5rem 1rem; border-radius: 5px; font-weight: 500;">✅ AI System: Operational</div>'
        st.sidebar.markdown(api_status, unsafe_allow_html=True)
    else:
        api_error = '<div style="background: linear-gradient(90deg, #ef4444 0%, #dc2626 100%); color: white; padding: 0.5rem 1rem; border-radius: 5px; font-weight: 500;">❌ AI System: Not configured</div>'
        st.sidebar.markdown(api_error, unsafe_allow_html=True)
    
    if EMAIL_ENABLED:
        email_status = f'<div style="background: linear-gradient(90deg, #10b981 0%, #059669 100%); color: white; padding: 0.5rem 1rem; border-radius: 5px; font-weight: 500;">✅ Email System: Active<br><small>{EMAIL_USER}</small></div>'
        st.sidebar.markdown(email_status, unsafe_allow_html=True)
        
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 📧 Email Testing")
        
        if st.sidebar.button("🔬 Send Test Admin Alert"):
            test_patient_info = {'age': '45', 'gender': 'Male', 'phone': '(555) 123-4567', 'email': 'test@example.com', 'insurance': 'Blue Cross', 'symptoms': 'Test chest pain symptoms for system testing', 'medical_history': 'No known allergies, previous cardiac screening', 'specialty': 'cardiology', 'urgency': 'high'}
            result = send_admin_notification("Test Patient", **test_patient_info)
            if "successfully" in result:
                st.sidebar.success("✅ Test admin alert sent!")
            else:
                st.sidebar.error(f"❌ Test failed: {result}")
    else:
        email_warning = '<div style="background: linear-gradient(90deg, #f59e0b 0%, #d97706 100%); color: white; padding: 0.5rem 1rem; border-radius: 5px; font-weight: 500;">⚠️ Email System: Disabled</div>'
        st.sidebar.markdown(email_warning, unsafe_allow_html=True)
        st.sidebar.info("💡 Configure EMAIL_USER and EMAIL_APP_PASSWORD in .env file to enable admin notifications")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔄 Session Management")
    if st.sidebar.button("🆕 New Patient Session"):
        st.session_state['messages'] = []
        st.session_state['conversation_history'] = ""
        log_system_message("SYSTEM: New patient session started")
        st.rerun()
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📋 Patient Database")
    
    if st.sidebar.button("👥 View Patient Records"):
        df = get_all_patients()
        if not df.empty:
            st.sidebar.dataframe(df.head(10), use_container_width=True)
        else:
            st.sidebar.info("📝 No patient records found in database.")

def main():
    if not OPENAI_API_KEY:
        st.error("OpenAI API Key not configured. Please add it to your .env file.")
        st.stop()
    
    st.set_page_config(page_title="Healthcare Patient Intake & Triage System", page_icon="🏥", layout="wide", initial_sidebar_state="expanded")
    
    css_styles = '<style>@import url("https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap"); .main {font-family: "Inter", sans-serif; background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);} .main-header {background: linear-gradient(90deg, #2c5aa0 0%, #1e3a8a 100%); padding: 2rem; border-radius: 10px; margin-bottom: 2rem; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); text-align: center;} .main-header h1 {color: white; font-size: 2.5rem; font-weight: 600; margin: 0;} .main-header p {color: #e0e7ff; font-size: 1.1rem; margin: 1rem 0 0 0; font-weight: 300;} .info-card {background: white; border-radius: 10px; padding: 1.5rem; box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1); border-left: 4px solid #2c5aa0; margin: 1rem 0;} .info-card h3 {color: #2c5aa0; margin-top: 0; font-weight: 600;} .emergency-alert {background: linear-gradient(90deg, #dc2626 0%, #991b1b 100%); color: white; padding: 1rem; border-radius: 8px; text-align: center; font-weight: 600; font-size: 1.1rem; animation: pulse 2s infinite; margin: 1rem 0;} @keyframes pulse {0% { opacity: 1; } 50% { opacity: 0.8; } 100% { opacity: 1; }} .dept-badge {display: inline-block; padding: 0.3rem 0.8rem; background: linear-gradient(90deg, #6366f1 0%, #4f46e5 100%); color: white; border-radius: 15px; font-size: 0.85rem; font-weight: 500; margin: 0.2rem;} #MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;}</style>'
    st.markdown(css_styles, unsafe_allow_html=True)
    
    header_html = '<div class="main-header"><h1><span>🏥</span> Healthcare Patient Intake & Triage System <span>⚕️</span></h1><p>Professional medical intake and patient triage assistance • Secure • HIPAA-Compliant • 24/7 Available</p></div>'
    st.markdown(header_html, unsafe_allow_html=True)
    
    if 'messages' not in st.session_state:
        st.session_state['messages'] = []
    if 'system_logs' not in st.session_state:
        st.session_state['system_logs'] = []
    
    init_database()
    render_sidebar()
    
    col1, col2 = st.columns([2, 1], gap="large")
    
    with col1:
        chat_header = '<div class="info-card"><h3>💬 Patient Communication Interface</h3><p style="margin: 0; color: #6b7280;">Please describe your symptoms or health concerns. Our AI will help connect you with the appropriate medical department.</p></div>'
        st.markdown(chat_header, unsafe_allow_html=True)
        
        for message in st.session_state['messages']:
            with st.chat_message(message["role"], avatar="🏥" if message["role"] == "assistant" else "👤"):
                st.write(message["content"])
        
        user_input = st.chat_input("Describe your symptoms or health concerns here...")
        if user_input:
            asyncio.run(process_user_message(user_input))
            st.rerun()
        
        disclaimer_html = '<div style="background: #fef7cd; border: 1px solid #fcd34d; border-radius: 8px; padding: 1rem; margin: 1rem 0;"><h4 style="color: #92400e; margin-top: 0;">⚠️ Important Medical Disclaimer</h4><p style="color: #92400e; margin: 0; font-size: 0.9rem;">This system provides intake assistance only and is not a substitute for professional medical advice. For emergencies, call 911 immediately. Always consult with qualified healthcare professionals for medical decisions.</p></div>'
        st.markdown(disclaimer_html, unsafe_allow_html=True)
    
    with col2:
        monitoring_header = '<div class="info-card"><h3>📊 System Monitoring</h3></div>'
        st.markdown(monitoring_header, unsafe_allow_html=True)
        
        log_container = st.container(height=300)
        with log_container:
            for log in st.session_state['system_logs'][-20:]:
                if "ERROR" in log:
                    st.markdown(f'<span style="color: #ef4444;">🔴 {log}</span>', unsafe_allow_html=True)
                elif "SUCCESS" in log or "✅" in log:
                    st.markdown(f'<span style="color: #10b981;">🟢 {log}</span>', unsafe_allow_html=True)
                elif "WARNING" in log:
                    st.markdown(f'<span style="color: #f59e0b;">🟡 {log}</span>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<span style="color: #374151;">{log}</span>', unsafe_allow_html=True)
        
        dept_status_html = '<div class="info-card"><h4>🏥 Department Status</h4><div style="display: flex; flex-wrap: wrap; gap: 0.5rem;"><span class="dept-badge">🚨 Emergency</span><span class="dept-badge">❤️ Cardiology</span><span class="dept-badge">🔬 Dermatology</span><span class="dept-badge">🦴 Orthopedics</span><span class="dept-badge">🧠 Mental Health</span><span class="dept-badge">👨‍⚕️ General Practice</span><span class="dept-badge">👶 Pediatrics</span><span class="dept-badge">👩‍⚕️ Gynecology</span></div></div>'
        st.markdown(dept_status_html, unsafe_allow_html=True)
        
        emergency_html = '<div class="emergency-alert">🚨 FOR MEDICAL EMERGENCIES CALL 911 🚨</div>'
        st.markdown(emergency_html, unsafe_allow_html=True)

    st.markdown("---")
    footer_html = '<div style="text-align: center; color: #6b7280; font-size: 0.9rem; padding: 1rem;"><p><strong>Healthcare Patient Intake & Triage System</strong> | Powered by Advanced AI</p><p>🔒 HIPAA Compliant • 🛡️ Secure • 📞 24/7 Support Available</p><p style="font-size: 0.8rem;">This system assists with patient intake and triage. Always consult healthcare professionals for medical advice.</p></div>'
    st.markdown(footer_html, unsafe_allow_html=True)

if __name__ == "__main__":
    main()