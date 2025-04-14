# propsaber/core/constants.py
"""
Define constants, global configurations, formatting strings.
Logging setup moved to app.py.
Extracted from the main script.
"""

import logging # Keep logging import for logger instance
from pathlib import Path
from typing import Dict

# It's good practice for modules to get their own logger
# NOTE: BasicConfig is called ONCE in app.py now.
logger = logging.getLogger(__name__)

# --- File Paths & Config ---
# SCENARIO_DIR: Directory to save/load scenario JSON files.
SCENARIO_DIR = Path("scenarios")
# FORWARD_CURVE_PATH: Path to the SOFR forward curve CSV file (relative to app.py).
# Note: This might be better handled by passing the path during initialization or loading.
FORWARD_CURVE_PATH = Path("Pensford_Forward_Curve.csv")
# FORWARD_CURVE_SD_COL_NAME: Column name in the CSV containing the +2 Standard Deviation rate data.
FORWARD_CURVE_SD_COL_NAME = "PlusTwoSD" # Make sure this matches your CSV

# --- Simulation Variables & Indices ---
# VAR_IDX: Mapping of variable names to their index in correlation matrices/shock arrays.
VAR_IDX: Dict[str, int] = {"Rent": 0, "OtherInc": 1, "Expense": 2, "ExitCap": 3, "Vacancy": 4}
# NUM_CORRELATED_VARS: Total number of variables included in the correlation structure.
NUM_CORRELATED_VARS: int = len(VAR_IDX)

# --- Numerical Constants ---
# FLOAT_ATOL: Absolute tolerance for floating-point comparisons (e.g., checking if a value is close to zero).
FLOAT_ATOL = 1e-9
# FLOAT_RTOL: Relative tolerance for floating-point comparisons.
FLOAT_RTOL = 1e-9
# MONTHS_PER_YEAR: Standard number of months in a year.
MONTHS_PER_YEAR: int = 12 # Still defined here for other modules

# --- Loan Types ---
LOAN_TYPE_IO = "Interest Only" # Still defined here for other modules
LOAN_TYPE_AMORT = "Amortizing" # Still defined here for other modules
LOAN_TYPES = [LOAN_TYPE_IO, LOAN_TYPE_AMORT] # Still defined here for other modules

# --- Formatting Constants ---
# FMT_CURRENCY_ZERO_DP: Format string for currency with zero decimal places (e.g., "$1,234").
FMT_CURRENCY_ZERO_DP = "%.0f"
# FMT_PERCENT_ONE_DP: Format string for percentages with one decimal place (e.g., "5.1").
FMT_PERCENT_ONE_DP = "%.1f"
# FMT_PERCENT_TWO_DP: Format string for percentages with two decimal places (e.g., "5.12").
FMT_PERCENT_TWO_DP = "%.2f"
# FMT_DECIMAL_TWO_DP: Format string for decimals with two decimal places (e.g., "0.12").
FMT_DECIMAL_TWO_DP = "%.2f"
# FMT_INTEGER: Format string for integers (e.g., "10").
FMT_INTEGER = "%d"

# --- Default Simulation Parameters ---
DEFAULT_HOLD_PERIOD = 10
DEFAULT_NUM_SIMULATIONS = 1000

# --- Add any other static constants extracted from your script ---
