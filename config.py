"""Configuration settings for the dashboard."""

# ==================== DATA SETTINGS ====================
DATA_FILE = 'emails.csv'
DEFAULT_SAMPLE_SIZE = 2000
MIN_SAMPLE_SIZE = 500
MAX_SAMPLE_SIZE = 10000
SAMPLE_STEP = 500

# ==================== SENTIMENT SETTINGS ====================
SENTIMENT_MODEL = 'vader'  # options: 'textblob', 'vader'
SENTIMENT_STRESSED_THRESHOLD = -0.1
SENTIMENT_HEALTHY_THRESHOLD = 0.1
SENTIMENT_SEVERE_STRESS_THRESHOLD = -0.2

# ==================== NETWORK SETTINGS ====================
DEFAULT_TOP_N_NODES = 50
MIN_TOP_N_NODES = 10
MAX_TOP_N_NODES = 100
TOP_N_STEP = 10

# Network visualization physics
NETWORK_GRAVITATIONAL_CONSTANT = -8000
NETWORK_SPRING_LENGTH = 150
NETWORK_SPRING_CONSTANT = 0.04
NETWORK_STABILIZATION_ITERATIONS = 100

# ==================== ANONYMIZATION SETTINGS ====================
ANONYMIZE_BY_DEFAULT = True
EXECUTIVE_THRESHOLD = 10  # Top N profiles labeled as Executives
MANAGER_THRESHOLD = 50    # Next N profiles labeled as Managers

# ==================== VISUALIZATION COLORS ====================
COLOR_STRESSED = "#d32f2f"      # Red
COLOR_NEUTRAL = "#757575"       # Gray
COLOR_HEALTHY = "#388e3c"       # Green
COLOR_PRIMARY = "#1976d2"       # Blue
COLOR_SECONDARY = "#455a64"     # Blue Gray

# Neutral backgrounds
GRADIENT_PRIMARY = "linear-gradient(135deg, #1976d2 0%, #1565c0 100%)"
GRADIENT_DANGER = "linear-gradient(135deg, #d32f2f 0%, #c62828 100%)"
GRADIENT_SUCCESS = "linear-gradient(135deg, #388e3c 0%, #2e7d32 100%)"
GRADIENT_WARNING = "linear-gradient(135deg, #f57c00 0%, #ef6c00 100%)"
GRADIENT_INFO = "linear-gradient(135deg, #0288d1 0%, #0277bd 100%)"

# ==================== UI SETTINGS ====================
PAGE_TITLE = "Enron Organizational Dynamics Monitor"
PAGE_ICON = "chart"
LAYOUT = "wide"

# Alert thresholds
HIGH_RISK_PERCENTAGE = 0.2  # 20% stressed emails triggers alert
HEALTHY_CULTURE_PERCENTAGE = 0.6  # 60% healthy emails shows success

# ==================== PERFORMANCE SETTINGS ====================
PROGRESS_UPDATE_FREQUENCY = 100  # Update progress bar every N rows
CACHE_TTL = 3600  # Cache time-to-live in seconds (1 hour)

# ==================== EXPORT SETTINGS ====================
REPORT_FILENAME_PREFIX = "organizational_intelligence_report"
RISK_REPORT_FILENAME = "risk_profiles_report.csv"

# ==================== PRIVACY SETTINGS ====================
SHOW_REAL_NAMES = False  # Set to True only for authorized users
ENABLE_NAME_SEARCH = False  # Allow reverse lookup (admin only)
LOG_ACCESS = True  # Log who accesses sensitive data
