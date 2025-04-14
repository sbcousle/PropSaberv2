# propsaber/core/utils.py
"""
Utility functions for the propsaber core simulation engine.
"""
import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, List, Optional
from functools import wraps

# Import necessary constants (relative import)
from .constants import NUM_CORRELATED_VARS, VAR_IDX, FLOAT_ATOL

logger = logging.getLogger(__name__)

def convert_to_internal(ui_val: Any, is_percentage_decimal: bool, scale_factor: float = 100.0) -> float:
    """
    Converts a UI input value (potentially string or number) to its internal float representation.
    Handles percentage conversion based on the flag.

    Args:
        ui_val: The value from the Streamlit UI widget.
        is_percentage_decimal: True if the UI value represents a percentage that should be
                               stored as a decimal internally (e.g., UI '5.0' -> internal 0.05).
                               False if the UI value should be stored directly as a float
                               (e.g., direct percentage points like growth rates, or currency).
        scale_factor: The factor to divide by if is_percentage_decimal is True (default 100.0).

    Returns:
        The converted float value, or np.nan if conversion fails.
    """
    if ui_val is None:
        return np.nan
    try:
        numeric_val = float(ui_val)
        if not np.isfinite(numeric_val):
             # Handle cases like float('inf')
             raise ValueError("Input value is not finite")
        # If it represents a percentage meant to be stored as a decimal
        if is_percentage_decimal:
            return numeric_val / scale_factor
        # Otherwise, return the float value directly
        else:
            return numeric_val
    except (ValueError, TypeError) as e:
        logger.warning(f"Could not convert UI value '{ui_val}' (type: {type(ui_val)}) to internal float: {e}")
        return np.nan

def get_correlation_matrix(inputs) -> np.ndarray:
    """
    Creates the correlation matrix based on SimulationInputs.

    Args:
        inputs: An object with correlation attributes (e.g., SimulationInputs instance).

    Returns:
        A NumPy array representing the correlation matrix. Returns identity if
        use_correlations is False or if the matrix is invalid.
    """
    if not getattr(inputs, 'use_correlations', False):
        logger.debug("Correlations not enabled, returning identity matrix.")
        return np.identity(NUM_CORRELATED_VARS)

    matrix = np.identity(NUM_CORRELATED_VARS)
    try:
        # Safely get correlation values using getattr with defaults
        corr_rent_expense = getattr(inputs, 'corr_rent_expense', 0.0)
        corr_rent_other_income = getattr(inputs, 'corr_rent_other_income', 0.0)
        corr_rent_vacancy = getattr(inputs, 'corr_rent_vacancy', 0.0)

        # Populate the matrix (symmetric)
        matrix[VAR_IDX["Rent"], VAR_IDX["Expense"]] = matrix[VAR_IDX["Expense"], VAR_IDX["Rent"]] = corr_rent_expense
        matrix[VAR_IDX["Rent"], VAR_IDX["OtherInc"]] = matrix[VAR_IDX["OtherInc"], VAR_IDX["Rent"]] = corr_rent_other_income
        matrix[VAR_IDX["Rent"], VAR_IDX["Vacancy"]] = matrix[VAR_IDX["Vacancy"], VAR_IDX["Rent"]] = corr_rent_vacancy
        # Add other correlations here if needed (e.g., Expense <-> OtherInc)

        # Validate positive semi-definiteness using Cholesky attempt
        np.linalg.cholesky(matrix)
        logger.debug(f"Generated Correlation Matrix:\n{matrix}")
        return matrix
    except np.linalg.LinAlgError:
        logger.warning("Correlation matrix not positive semi-definite. Using identity matrix instead.")
        return np.identity(NUM_CORRELATED_VARS)
    except Exception as e:
        logger.error(f"Error creating correlation matrix: {e}", exc_info=True)
        return np.identity(NUM_CORRELATED_VARS) # Fallback to identity on any error


def generate_correlated_shocks(L_matrix: np.ndarray, size: int) -> Optional[np.ndarray]:
    """
    Generates correlated standard normal shocks using Cholesky decomposition matrix.

    Args:
        L_matrix: The lower triangular matrix from Cholesky decomposition (L such that L*L.T = CorrMatrix).
        size: The number of shock vectors (time periods or simulations) to generate.

    Returns:
        A NumPy array of shape (size, NUM_CORRELATED_VARS) with correlated shocks,
        or None if input is invalid.
    """
    if L_matrix is None or L_matrix.shape[0] != NUM_CORRELATED_VARS or L_matrix.shape[1] != NUM_CORRELATED_VARS:
         logger.error(f"Invalid L_matrix provided to generate_correlated_shocks. Shape: {L_matrix.shape if L_matrix is not None else 'None'}")
         return None
    try:
        # Generate independent standard normal shocks
        Z = np.random.normal(0, 1, (size, NUM_CORRELATED_VARS))
        # Apply the Cholesky matrix transformation: CorrelatedShocks = Z * L.T
        correlated_shocks = Z @ L_matrix.T
        return correlated_shocks
    except Exception as e:
        logger.error(f"Error generating correlated shocks: {e}", exc_info=True)
        return None


def get_valid_paths(
    results_list: List[Dict[str, Any]],
    key: str,
    hold_period_actual: int
    ) -> List[List[float]]:
    """
    Extracts valid, finite numerical paths for a given key from a list of simulation results.

    Args:
        results_list: List of dictionaries, where each dict is a single simulation result.
        key: The key corresponding to the list of annual values within each result dict.
        hold_period_actual: The expected length of the annual data list.

    Returns:
        A list of lists, where each inner list is a valid path for the specified key.
    """
    paths = []
    if not isinstance(results_list, list):
        logger.warning(f"get_valid_paths: results_list is not a list (type: {type(results_list)}) for key '{key}'.")
        return paths

    for r_idx, r in enumerate(results_list):
        if not isinstance(r, dict):
            logger.debug(f"Skipping invalid result item (not a dict) at index {r_idx} for key '{key}'.")
            continue

        path = r.get(key)
        if (isinstance(path, list) and
            len(path) == hold_period_actual and
            all(isinstance(x, (int, float)) and np.isfinite(x) for x in path)):
            paths.append(path)
        else:
            logger.debug(f"Skipping invalid path for key '{key}' in run result #{r_idx}: Length={len(path) if isinstance(path, list) else 'N/A'} vs Expected={hold_period_actual}, Data={path}")
    logger.debug(f"Found {len(paths)} valid paths for key '{key}'.")
    return paths

def simulation_error_handler(func):
    """
    Decorator to consistently handle and log errors from simulation-related functions.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {e}", exc_info=True)
            return None  # or another appropriate default value based on your function logic
    return wrapper
