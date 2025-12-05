import streamlit as st
from config import (
    EXECUTIVE_THRESHOLD,
    MANAGER_THRESHOLD
)


@st.cache_data
def create_anonymization_map(names):
    """Map real names to anonymized IDs (Executive/Manager/Employee hierarchy)."""
    unique_names = sorted(set(names))
    anonymization_map = {}
    
    for idx, name in enumerate(unique_names, 1):
        if idx <= EXECUTIVE_THRESHOLD:
            anon_id = f"Executive-{idx:02d}"
        elif idx <= MANAGER_THRESHOLD:
            anon_id = f"Manager-{idx:03d}"
        else:
            anon_id = f"Employee-{idx:04d}"
        
        anonymization_map[name] = anon_id
    
    return anonymization_map


def anonymize_name(name, anonymization_map):
    """Anonymize a single name using the mapping."""
    return anonymization_map.get(name, f"Unknown-{hash(name) % 10000:04d}")


def apply_anonymization(df, anonymization_map):
    """Add anonymized sender and receiver columns to dataframe."""
    df['sender_anon'] = df['sender_clean'].apply(lambda x: anonymize_name(x, anonymization_map))
    df['receiver_anon'] = df['receiver_clean'].apply(lambda x: anonymize_name(x, anonymization_map))
    df.attrs['anonymization_map'] = anonymization_map
    
    return df


def get_real_name(anon_id, anonymization_map, authorized=False):
    """Reverse lookup from anonymized ID to real name (authorized users only)."""
    if not authorized:
        return anon_id
    
    reverse_map = {v: k for k, v in anonymization_map.items()}
    return reverse_map.get(anon_id, anon_id)
