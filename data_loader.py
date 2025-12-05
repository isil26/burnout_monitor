import pandas as pd
import streamlit as st
import re
from config import (
    DATA_FILE,
    PROGRESS_UPDATE_FREQUENCY
)


@st.cache_data(show_spinner=False)
def parse_email(raw_message):
    """Parse raw email text using regex to extract sender, receiver, date, subject, and body."""
    try:
        # x-from/x-to are more reliable than from/to headers
        sender_match = re.search(r'X-From:\s*(.+?)(?:\n|$)', raw_message, re.IGNORECASE)
        sender = sender_match.group(1).strip() if sender_match else None
        
        receiver_match = re.search(r'X-To:\s*(.+?)(?:\n|$)', raw_message, re.IGNORECASE)
        receiver = receiver_match.group(1).strip() if receiver_match else None
        
        date_match = re.search(r'Date:\s*(.+?)(?:\n|$)', raw_message)
        date_str = date_match.group(1).strip() if date_match else None
        
        parsed_date = None
        if date_str:
            try:
                date_str_clean = re.sub(r'\s*\([^)]*\)$', '', date_str)
                date_str_clean = re.sub(r'\s+\(.*$', '', date_str_clean)
                parsed_date = pd.to_datetime(date_str_clean, errors='coerce')
            except:
                pass
        
        subject_match = re.search(r'Subject:\s*(.+?)(?:\n|$)', raw_message)
        subject = subject_match.group(1).strip() if subject_match else ""
        
        body_match = re.search(r'X-FileName:.*?\n\s*(.+)', raw_message, re.DOTALL)
        body = body_match.group(1).strip() if body_match else ""
        
        return {
            'sender': sender,
            'receiver': receiver,
            'date': parsed_date,
            'subject': subject,
            'body': body
        }
    except Exception as e:
        return {
            'sender': None,
            'receiver': None,
            'date': None,
            'subject': "",
            'body': ""
        }


def clean_name(name):
    """Remove email addresses in angle brackets from name string."""
    if not name:
        return ""
    return re.sub(r'<.*?>', '', str(name)).strip()


@st.cache_data(show_spinner=False)
def load_and_parse_emails(sample_size=2000, load_full=False):
    """Load Enron emails from CSV and parse with regex."""
    try:
        if load_full:
            df = pd.read_csv(DATA_FILE)
            st.info(f" Loading full dataset... This may take a few minutes.")
        else:
            df = pd.read_csv(DATA_FILE, nrows=sample_size)
        
        parsed_emails = []
        progress_bar = st.progress(0)
        total_rows = len(df)
        
        for idx, row in df.iterrows():
            if idx % PROGRESS_UPDATE_FREQUENCY == 0:
                progress_bar.progress(min(idx / total_rows, 1.0))
            
            parsed = parse_email(row['message'])
            parsed_emails.append(parsed)
        
        progress_bar.empty()
        
        parsed_df = pd.DataFrame(parsed_emails)
        parsed_df = parsed_df.dropna(subset=['sender', 'receiver'])
        parsed_df['date'] = pd.to_datetime(parsed_df['date'], errors='coerce')
        parsed_df['sender_clean'] = parsed_df['sender'].apply(clean_name)
        parsed_df['receiver_clean'] = parsed_df['receiver'].apply(clean_name)
        
        return parsed_df
    
    except FileNotFoundError:
        st.error(f" Error: {DATA_FILE} not found. Please ensure the file is in the project directory.")
        return None
    except Exception as e:
        st.error(f" Error loading data: {str(e)}")
        return None


def apply_date_filter(df, date_range):
    """Filter dataframe by date range."""
    if len(date_range) == 2:
        mask = (df['date'].dt.date >= date_range[0]) & (df['date'].dt.date <= date_range[1])
        return df[mask]
    return df
