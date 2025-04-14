# propsaber/data/forward_curve.py
"""
Functions for loading and processing the SOFR forward curve data from CSV.
Extracted from the main script.
"""

import pandas as pd
import streamlit as st # Used for st.error/st.warning
import logging
from typing import Dict, Tuple, Optional
from pathlib import Path # Use pathlib for path handling
from propsaber.core.utils import simulation_error_handler

# Import constants from the core module using relative import
from ..core.constants import FORWARD_CURVE_SD_COL_NAME

logger = logging.getLogger(__name__)

# Note: Removed global variables FORWARD_SOFR_CURVE, FORWARD_SOFR_STD_DEV.
# The loading function now returns the data.

# Use st.cache_data for caching data loading results in Streamlit apps
@simulation_error_handler
@st.cache_data(ttl=3600) # Cache for 1 hour
def load_forward_curve_and_std_dev(curve_file_path: Path) -> Tuple[Dict[int, float], Dict[int, float]]:
    """
    Loads forward SOFR curve AND calculates annual standard deviation from +2SD column in CSV.

    Args:
        curve_file_path: The Path object pointing to the CSV file.

    Returns:
        A tuple containing two dictionaries:
        1. Forward SOFR curve {year: rate_decimal}
        2. Forward SOFR standard deviation {year: sd_decimal}
        Returns empty dictionaries ({}, {}) if loading or processing fails.
    """
    forward_sofr_curve: Dict[int, float] = {}
    forward_sofr_std_dev: Dict[int, float] = {}

    if not curve_file_path.is_file():
        st.error(f"Forward curve file not found: {curve_file_path}. Please ensure it exists.")
        logger.error(f"Forward curve file missing: {curve_file_path}")
        return forward_sofr_curve, forward_sofr_std_dev

    try:
        forward_df = pd.read_csv(curve_file_path)
        required_columns = ['Year', 'AvgForwardRate', FORWARD_CURVE_SD_COL_NAME]

        # --- Validation ---
        if not all(col in forward_df.columns for col in required_columns):
            missing_cols = [col for col in required_columns if col not in forward_df.columns]
            raise ValueError(f"CSV must contain required columns. Missing: {', '.join(missing_cols)}")

        # --- Data Cleaning & Type Conversion ---
        # Convert relevant columns to numeric, errors='coerce' will turn non-numeric into NaN
        numeric_cols = ['AvgForwardRate', FORWARD_CURVE_SD_COL_NAME]
        for col in numeric_cols:
            original_type = forward_df[col].dtype
            forward_df[col] = pd.to_numeric(forward_df[col], errors='coerce')
            if forward_df[col].isnull().any():
                logger.warning(f"Column '{col}' in {curve_file_path} contained non-numeric values which were ignored (originally type {original_type}).")

        # Drop rows with NaN in essential columns after conversion
        forward_df.dropna(subset=required_columns, inplace=True)
        if forward_df.empty:
            raise ValueError("No valid numeric data found in required columns after cleaning.")

        # Ensure Year is integer
        try:
            forward_df['Year'] = forward_df['Year'].astype(int)
        except (ValueError, TypeError):
            raise ValueError("Could not convert 'Year' column to integer.")

        # --- Calculations ---
        # Calculate Standard Deviation from the +2SD column
        # SD = (+2SD Rate - Average Rate) / 2
        # Convert result to decimal (divide by 100)
        forward_df['CalculatedSD_Decimal'] = (forward_df[FORWARD_CURVE_SD_COL_NAME] - forward_df['AvgForwardRate']) / 2.0 / 100.0
        # Ensure SD is non-negative
        forward_df['CalculatedSD_Decimal'] = forward_df['CalculatedSD_Decimal'].apply(lambda x: max(0.0, x))

        # Convert Average Rate to decimal
        forward_df['AvgForwardRate_Decimal'] = forward_df['AvgForwardRate'] / 100.0

        # --- Handle Duplicates and Create Dictionaries ---
        if forward_df['Year'].duplicated().any():
            logger.warning(f"Duplicate years found in curve data ({curve_file_path}). Using the last entry for each year.")
            forward_df = forward_df.drop_duplicates(subset='Year', keep='last')

        # Set index and create dictionaries
        forward_df.set_index('Year', inplace=True)
        forward_sofr_curve = forward_df['AvgForwardRate_Decimal'].to_dict()
        forward_sofr_std_dev = forward_df['CalculatedSD_Decimal'].to_dict()

        logger.info(f"Successfully loaded forward curve and calculated std devs for {len(forward_sofr_curve)} years from {curve_file_path.name}.")
        logger.debug(f"First 5 SOFR rates: {dict(list(forward_sofr_curve.items())[:5])}")
        logger.debug(f"First 5 SOFR std devs: {dict(list(forward_sofr_std_dev.items())[:5])}")

        return forward_sofr_curve, forward_sofr_std_dev

    except FileNotFoundError: # Should be caught by is_file() check, but included for robustness
        st.error(f"Forward curve file not found at {curve_file_path}.")
        logger.error(f"Forward curve file missing: {curve_file_path}")
        return {}, {}
    except ValueError as ve:
        st.error(f"Error processing curve file '{curve_file_path.name}': {ve}")
        logger.error(f"Curve file processing error ({curve_file_path.name}): {ve}")
        return {}, {}
    except pd.errors.EmptyDataError:
         st.error(f"Forward curve file '{curve_file_path.name}' is empty.")
         logger.error(f"Forward curve file is empty: {curve_file_path}")
         return {}, {}
    except Exception as e:
        st.error(f"An unexpected error occurred loading forward curve/std dev from '{curve_file_path.name}': {e}")
        logger.error(f"Forward curve/std dev load error ({curve_file_path.name}): {e}", exc_info=True)
        return {}, {}

# --- Optional: Helper function to get rate for a specific year ---
@simulation_error_handler
def get_rate_for_year(year: int, curve: Dict[int, float]) -> Optional[float]:
    """
    Retrieves the rate for a specific year from a curve dictionary.
    Handles missing years by returning the rate from the latest available year.

    Args:
        year: The target year.
        curve: The dictionary representing the rate curve {year: rate}.

    Returns:
        The rate for the year, or the rate of the latest year if the target year
        is beyond the curve, or None if the curve is empty.
    """
    if not curve:
        return None
    if year in curve:
        return curve[year]
    else:
        # Find the latest year in the curve and return its rate as a fallback
        try:
            latest_year = max(curve.keys())
            return curve.get(latest_year)
        except ValueError: # Should not happen if curve is not empty, but safe check
            return None