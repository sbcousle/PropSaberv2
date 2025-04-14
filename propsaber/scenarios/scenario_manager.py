# propsaber/scenarios/scenario_manager.py
"""
Functions for saving, loading, and listing simulation input scenarios stored as JSON files.
Extracted and adapted from the main script.
"""

import json
import os
import streamlit as st # Used for st.success/st.error/st.warning
import logging
import numpy as np # Needed for serializing numpy types
from typing import Dict, Any, Optional, List
from pathlib import Path
from propsaber.core.utils import simulation_error_handler

# Import constants from the core module using relative import
from ..core.constants import SCENARIO_DIR # Use the defined scenario directory path

logger = logging.getLogger(__name__)

def _ensure_scenario_dir_exists():
    """Creates the scenario directory if it doesn't exist."""
    try:
        SCENARIO_DIR.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        st.error(f"Could not create scenario directory '{SCENARIO_DIR}': {e}")
        logger.error(f"Failed to create scenario directory '{SCENARIO_DIR}': {e}")
        # Depending on desired behavior, might raise the error or return False

def _numpy_converter(obj):
    """Helper function to convert NumPy types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    # Add other NumPy types if needed
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

@simulation_error_handler
def save_scenario(inputs_dict: Dict[str, Any], filename: str):
    """
    Saves the provided simulation inputs dictionary to a JSON file.

    Args:
        inputs_dict: A dictionary containing the simulation inputs (e.g., from SimulationInputs.to_dict()).
        filename: The desired filename (e.g., "my_scenario.json"). It will be saved in the SCENARIO_DIR.
    """
    _ensure_scenario_dir_exists()
    filepath = SCENARIO_DIR / filename # Use pathlib for path joining

    try:
        # Use the custom converter to handle NumPy types during serialization
        with open(filepath, 'w') as f:
            json.dump({"inputs": inputs_dict}, f, indent=4, default=_numpy_converter)
        st.success(f"Scenario inputs saved to `{filepath}`")
        logger.info(f"Scenario saved successfully: {filepath}")
    except TypeError as te:
         st.error(f"Error saving scenario '{filename}': Data type issue. {te}")
         logger.error(f"Save Scenario TypeError for '{filename}': {te}", exc_info=True)
    except Exception as e:
        st.error(f"Error saving scenario '{filename}': {e}")
        logger.error(f"Save Scenario Error for '{filename}': {e}", exc_info=True)

@simulation_error_handler
def load_scenario(filename: str) -> Optional[Dict[str, Any]]:
    """
    Loads simulation inputs from a specified JSON scenario file.

    Args:
        filename: The name of the scenario file (e.g., "my_scenario.json") located in SCENARIO_DIR.

    Returns:
        A dictionary containing the loaded simulation inputs, or None if loading fails.
    """
    filepath = SCENARIO_DIR / filename
    if not filepath.is_file():
        # Don't show error if file just doesn't exist (might be expected)
        st.info(f"Scenario file not found: `{filepath}`")
        logger.info(f"Scenario file not found during load attempt: {filepath}")
        return None

    try:
        with open(filepath, 'r') as f:
            loaded_data = json.load(f)

        # Basic validation of the loaded structure
        if "inputs" in loaded_data and isinstance(loaded_data["inputs"], dict):
            logger.info(f"Scenario loaded successfully: {filepath}")
            return loaded_data["inputs"]
        else:
            st.error(f"Invalid format in `{filepath}`. Expected a JSON object with an 'inputs' key containing the scenario data.")
            logger.error(f"Invalid format in scenario file: {filepath}")
            return None
    except json.JSONDecodeError:
        st.error(f"Error decoding JSON from `{filepath}`. File might be corrupted.")
        logger.error(f"JSONDecodeError loading scenario: {filepath}", exc_info=True)
        return None
    except Exception as e:
        st.error(f"Error loading scenario '{filename}': {e}")
        logger.error(f"Load Scenario Error for '{filename}': {e}", exc_info=True)
        return None

@simulation_error_handler
def list_saved_scenarios() -> List[str]:
    """
    Lists the names of saved scenario files (with .json extension).

    Returns:
        A sorted list of scenario filenames found in the SCENARIO_DIR.
        Returns an empty list if the directory doesn't exist or contains no .json files.
    """
    _ensure_scenario_dir_exists() # Ensure directory exists before listing
    try:
        # List only .json files within the scenario directory
        scenario_files = sorted([f.name for f in SCENARIO_DIR.iterdir() if f.is_file() and f.suffix == ".json"])
        return scenario_files
    except FileNotFoundError:
        # This might happen if _ensure_scenario_dir_exists failed silently or dir was deleted
        logger.warning(f"Scenario directory '{SCENARIO_DIR}' not found when listing scenarios.")
        return []
    except Exception as e:
        st.error(f"Error listing scenarios: {e}")
        logger.error(f"Error listing scenarios in '{SCENARIO_DIR}': {e}", exc_info=True)
        return []
