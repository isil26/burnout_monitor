"""CSS styling and HTML components."""

from config import (
    COLOR_PRIMARY,
    COLOR_SECONDARY,
    GRADIENT_PRIMARY,
    GRADIENT_DANGER,
    GRADIENT_SUCCESS,
    GRADIENT_WARNING,
    GRADIENT_INFO
)


def get_custom_css():
    """Return custom CSS styling."""
    return """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap');
    
    * {
        font-family: 'Roboto', sans-serif;
    }
    
    .main-header {
        font-size: 2.5rem;
        font-weight: 500;
        color: #212121;
        text-align: center;
        margin-bottom: 0.5rem;
        letter-spacing: -0.5px;
    }
    
    .subtitle {
        text-align: center;
        color: #616161;
        font-size: 1rem;
        margin-bottom: 2rem;
        font-weight: 300;
    }
    
    .metric-card {
        background: #ffffff;
        padding: 1.5rem;
        border-radius: 4px;
        border: 1px solid #e0e0e0;
        border-left: 4px solid """ + COLOR_PRIMARY + """;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.08);
        transition: box-shadow 0.2s ease;
    }
    
    .metric-card:hover {
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.12);
    }
    
    .toxic-warning {
        background: #ffffff;
        border-left: 4px solid #d32f2f;
        border: 1px solid #ffcdd2;
    }
    
    .healthy-indicator {
        background: #ffffff;
        border-left: 4px solid #388e3c;
        border: 1px solid #c8e6c9;
    }
    
    .neutral-indicator {
        background: #ffffff;
        border-left: 4px solid #757575;
        border: 1px solid #e0e0e0;
    }
    
    .stat-badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 4px;
        font-weight: 500;
        font-size: 0.875rem;
        margin: 0.25rem;
        border: 1px solid;
    }
    
    .stat-badge-danger {
        background: #ffebee;
        color: #c62828;
        border-color: #ef9a9a;
    }
    
    .stat-badge-success {
        background: #e8f5e9;
        color: #2e7d32;
        border-color: #a5d6a7;
    }
    
    .stat-badge-warning {
        background: #fff3e0;
        color: #e65100;
        border-color: #ffcc80;
    }
    
    .stat-badge-info {
        background: #e3f2fd;
        color: #1565c0;
        border-color: #90caf9;
    }
    
    .alert-box {
        padding: 1rem 1.5rem;
        border-radius: 4px;
        margin: 1rem 0;
        border-left: 4px solid;
        background: #ffffff;
    }
    
    .alert-danger {
        background-color: #ffebee;
        border-color: #d32f2f;
        color: #b71c1c;
    }
    
    .alert-success {
        background-color: #e8f5e9;
        border-color: #388e3c;
        color: #1b5e20;
    }
    
    .alert-info {
        background-color: #e3f2fd;
        border-color: #1976d2;
        color: #0d47a1;
    }
    
    .privacy-badge {
        display: inline-flex;
        align-items: center;
        padding: 0.5rem 1rem;
        background: #455a64;
        color: white;
        border-radius: 4px;
        font-size: 0.85rem;
        font-weight: 400;
        margin: 0.5rem 0;
    }
    
    .section-header {
        font-size: 1.5rem;
        font-weight: 500;
        color: #212121;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid """ + COLOR_PRIMARY + """;
    }
    
    [data-testid="stSidebar"] {
        background: #fafafa;
    }
    
    [data-testid="stSidebar"] * {
        color: #212121 !important;
    }
    
    [data-testid="stSidebar"] h2 {
        color: #1976d2 !important;
    }
    
    .stButton>button {
        background: """ + COLOR_PRIMARY + """;
        color: white;
        border: none;
        border-radius: 4px;
        padding: 0.5rem 1.5rem;
        font-weight: 500;
        transition: background 0.2s ease;
    }
    
    .stButton>button:hover {
        background: """ + COLOR_SECONDARY + """;
    }
    
    .dataframe {
        border-radius: 4px;
        overflow: hidden;
    }
</style>
"""


def create_privacy_badge():
    """
    Create HTML for privacy compliance badge.
    
    Returns:
        str: HTML code for privacy badge
    """
    return '''
    <div style="text-align: center;">
        <span class="privacy-badge">
            🔒 All employee data anonymized • GDPR Compliant • Secure Analytics
        </span>
    </div>
    '''


def create_alert_box(message, alert_type="info"):
    """
    Create styled alert box.
    
    Args:
        message (str): Alert message
        alert_type (str): Type of alert (danger, success, info)
        
    Returns:
        str: HTML code for alert box
    """
    return f'''
    <div class="alert-box alert-{alert_type}">
        {message}
    </div>
    '''


def create_stat_badge(label, value, badge_type="info"):
    """
    Create styled stat badge.
    
    Args:
        label (str): Badge label
        value (str): Badge value
        badge_type (str): Type of badge (danger, success, warning, info)
        
    Returns:
        str: HTML code for stat badge
    """
    return f'<span class="stat-badge stat-badge-{badge_type}">{label}: {value}</span>'


def create_section_header(text):
    """
    Create styled section header.
    
    Args:
        text (str): Header text
        
    Returns:
        str: HTML code for section header
    """
    return f'<div class="section-header">{text}</div>'
