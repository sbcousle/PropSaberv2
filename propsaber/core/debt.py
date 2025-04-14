# propsaber/core/debt.py
"""
Contains functions specifically related to calculating debt service,
handling both fixed and variable rate loans.

MODIFIED: calculate_debt_service now accepts pre-calculated monthly_payment
for amortizing loans instead of recalculating based on remaining term.
"""

import numpy as np
import numpy_financial as npf
import logging
from typing import Dict, Tuple, Optional
import random
from .utils import simulation_error_handler

# Import constants from the current package
from .constants import LOAN_TYPE_IO, LOAN_TYPE_AMORT, MONTHS_PER_YEAR, FLOAT_ATOL

logger = logging.getLogger(__name__) # Get logger for this module

# --- Helper Function for Variable Rate Components ---
@simulation_error_handler
def _calculate_variable_rate_components(
    year: int,
    forward_curve: Dict[int, float],
    std_dev_curve: Dict[int, float],
    volatility_scalar: float,
    prev_volatile_sofr_comp: Optional[float],
    rate_persistence_phi: float, # AR(1) persistence factor for rate component
    sofr_floor: float,
    sofr_spread: float
) -> Tuple[float, float]:
    """
    Calculates the variable effective rate and its underlying volatile component for a given year.
    This isolates the core logic of the variable rate simulation.
    (Implementation remains the same as provided in the original user code)
    """
    effective_rate = np.nan
    current_volatile_sofr_comp = np.nan # This is the value *before* floor/spread

    if not forward_curve or not std_dev_curve:
        logger.error(f"Variable rate calculation failed for Year {year}: Missing forward curve or std dev data. Forward Curve Keys: {list(forward_curve.keys()) if forward_curve else 'None'}, Std Dev Keys: {list(std_dev_curve.keys()) if std_dev_curve else 'None'}")
        return np.nan, np.nan

    try:
        max_curve_year = max(forward_curve.keys()) if forward_curve else 1
        fallback_rate = forward_curve.get(max_curve_year, 0.0)
        sofr_rate = forward_curve.get(year, fallback_rate)
        logger.debug(f"Year {year} (Var Helper): Retrieved sofr_rate = {sofr_rate}")
        if not np.isfinite(sofr_rate):
             logger.error(f"Year {year} (Var Helper): Retrieved non-finite sofr_rate: {sofr_rate}. Check forward_curve data and fallback logic.")
             return np.nan, np.nan

        if sofr_floor > 1.0: adjusted_sofr_floor = sofr_floor / 100.0
        elif sofr_floor < 0: adjusted_sofr_floor = 0.0
        else: adjusted_sofr_floor = sofr_floor

        max_sd_year = max(std_dev_curve.keys()) if std_dev_curve else 1
        fallback_sd = std_dev_curve.get(max_sd_year, 0.0)
        current_year_sd = std_dev_curve.get(year, fallback_sd)
        logger.debug(f"Year {year} (Var Helper): Retrieved current_year_sd = {current_year_sd}")
        if not np.isfinite(current_year_sd):
             logger.error(f"Year {year} (Var Helper): Retrieved non-finite current_year_sd: {current_year_sd}. Check std_dev_curve data and fallback logic.")
             return np.nan, np.nan

        scaled_sd = max(0.0, current_year_sd * volatility_scalar)
        try:
            # Use standard normal for shock generation before scaling by sd
            shock = random.gauss(0, 1) # Standard normal shock Z
            scaled_shock = shock * scaled_sd # sigma * Z
        except ValueError as ve:
            logger.error(f"Year {year} (Var Helper): Error generating shock with np.random.normal(0, {scaled_sd}): {ve}")
            return np.nan, np.nan

        logger.debug(f"Year {year} (Var Helper): volatility_scalar={volatility_scalar}, scaled_sd={scaled_sd}, shock={shock}")
        if not np.isfinite(scaled_shock):
             logger.error(f"Year {year} (Var Helper): Calculated non-finite shock: {scaled_shock}. Check scaled_sd calculation.")
             return np.nan, np.nan

        base_volatile_target = sofr_rate + scaled_shock
        logger.debug(f"Year {year} (Var Helper): base_volatile_target = {sofr_rate} + {scaled_shock} = {base_volatile_target}")
        if not np.isfinite(base_volatile_target):
             logger.error(f"Year {year} (Var Helper): Calculated non-finite base_volatile_target: {base_volatile_target}.")
             # Let persistence logic handle it

        phi = max(0.0, min(0.999, rate_persistence_phi)) # Clamp persistence factor
        if prev_volatile_sofr_comp is None or not np.isfinite(prev_volatile_sofr_comp) or abs(phi) < FLOAT_ATOL:
            current_volatile_sofr_comp = base_volatile_target
            logger.debug(f"Year {year} (Var Helper): Using base target for current_volatile_sofr_comp (no persistence or first year). Value = {current_volatile_sofr_comp}")
        else:
             if not np.isfinite(base_volatile_target):
                 current_volatile_sofr_comp = phi * prev_volatile_sofr_comp # Degenerate case if target fails
                 logger.warning(f"Year {year} (Var Helper): Base target non-finite, using only persistence term.")
             else:
                 current_volatile_sofr_comp = (phi * prev_volatile_sofr_comp +
                                               (1.0 - phi) * base_volatile_target)
             logger.debug(f"Year {year} (Var Helper): Applied persistence. phi={phi}, prev_comp={prev_volatile_sofr_comp}, target={base_volatile_target}. Result current_volatile_sofr_comp = {current_volatile_sofr_comp}")


        if not np.isfinite(current_volatile_sofr_comp):
             logger.error(f"Year {year} (Var Helper): current_volatile_sofr_comp became non-finite after persistence calculation: {current_volatile_sofr_comp}")
             return np.nan, np.nan

        floored_rate_component = max(current_volatile_sofr_comp, adjusted_sofr_floor)
        logger.debug(f"Year {year} (Var Helper): Applied floor. Max({current_volatile_sofr_comp=}, {adjusted_sofr_floor=}) = {floored_rate_component=}")

        effective_rate = floored_rate_component + sofr_spread
        logger.debug(f"Year {year} (Var Helper): Added spread. {floored_rate_component=} + {sofr_spread=} = {effective_rate=}")

        if not np.isfinite(effective_rate) or not np.isfinite(current_volatile_sofr_comp):
            logger.error(f"Variable rate component calculation resulted in NaN for Year {year} (Final Check). Rate={effective_rate}, Comp={current_volatile_sofr_comp}")
            return np.nan, np.nan

        return effective_rate, current_volatile_sofr_comp

    except Exception as e:
        logger.error(f"Error during variable rate component calculation Year {year}: {e}", exc_info=True)
        return np.nan, np.nan


# --- Main Debt Service Calculation Function ---
@simulation_error_handler
def calculate_debt_service(
    # --- Loan Terms (Potentially Updated by Refi) ---
    current_loan_type: str,
    # current_loan_amount: float, # No longer needed for pmt calc inside
    current_interest_rate: float, # Fixed rate OR placeholder if variable
    # current_loan_term_yrs: int, # No longer needed for pmt calc inside
    current_is_variable_rate: bool,
    # --- Current State ---
    current_balance: float,
    monthly_payment: float, # <<< ADDED: Pass pre-calculated constant payment
    year: int, # Current simulation year (1-based)
    # --- Variable Rate Parameters (Original inputs, used only if current_is_variable_rate is True) ---
    sofr_spread: float,
    forward_curve: Dict[int, float],
    std_dev_curve: Dict[int, float],
    sofr_floor: float,
    rate_persistence_phi: float,
    volatility_scalar: float,
    prev_volatile_sofr_comp: Optional[float] # Previous year's component
    ) -> Tuple[float, float, float, float, float]:
    """
    Calculates annual debt service (interest, principal, ending balance) and the effective interest rate.
    Uses the *current* loan terms passed.
    MODIFIED: Accepts pre-calculated monthly_payment for amortizing loans.
    """
    # Removed current_loan_amount, current_loan_term_yrs from log msg as they aren't direct inputs now
    logger.debug(f"Calculating debt service for Year {year}: is_variable={current_is_variable_rate}, loan_type={current_loan_type}, start_balance={current_balance:.2f}, monthly_pmt={monthly_payment:.2f}")

    # Removed check for current_loan_amount
    if current_balance <= FLOAT_ATOL:
        logger.debug(f"Year {year}: No loan or fully paid off (Balance: {current_balance:.2f}). Returning zeros.")
        return 0.0, 0.0, 0.0, 0.0, np.nan

    effective_rate = np.nan
    current_volatile_sofr_comp_this_year = np.nan # Initialize as NaN
    original_loan_type = current_loan_type # Store original type before potential override

    if current_is_variable_rate:
        logger.debug(f"Year {year}: Calling _calculate_variable_rate_components...")
        effective_rate, current_volatile_sofr_comp_this_year = _calculate_variable_rate_components(
            year=year, forward_curve=forward_curve, std_dev_curve=std_dev_curve,
            volatility_scalar=volatility_scalar, prev_volatile_sofr_comp=prev_volatile_sofr_comp,
            rate_persistence_phi=rate_persistence_phi, sofr_floor=sofr_floor, sofr_spread=sofr_spread
        )
        # Force IO for variable rate in this model version
        current_loan_type = LOAN_TYPE_IO
        logger.debug(f"Year {year}: Variable rate calculated. EffectiveRate={effective_rate:.4f}, VolatileComp={current_volatile_sofr_comp_this_year:.4f}. Forced LoanType={current_loan_type}")
    else: # Fixed rate case
        logger.debug(f"Year {year}: Using fixed rate: {current_interest_rate:.4f}")
        effective_rate = current_interest_rate
        # current_volatile_sofr_comp_this_year remains np.nan

    if not np.isfinite(effective_rate):
        logger.error(f"Effective rate calculation failed or resulted in NaN ({effective_rate}) for Year {year}. Cannot calculate debt service.")
        return np.nan, np.nan, np.nan, np.nan, np.nan

    interest_payment_yr = np.nan
    principal_payment_yr = np.nan
    ending_balance = current_balance

    try:
        if current_loan_type == LOAN_TYPE_IO:
            interest_payment_yr = current_balance * effective_rate
            principal_payment_yr = 0.0
            # Ending balance remains current_balance for IO
            ending_balance = current_balance
            logger.debug(f"Year {year} (IO): Interest={interest_payment_yr:.2f}, Principal={principal_payment_yr:.2f}, EndBalance={ending_balance:.2f}")

        elif current_loan_type == LOAN_TYPE_AMORT:
            # This path is only taken if fixed rate AND Amortizing type
            # periods = current_loan_term_yrs * MONTHS_PER_YEAR # Removed - not needed for pmt calc
            rate_monthly = effective_rate / MONTHS_PER_YEAR
            # payment_basis_loan_amount = current_loan_amount # Removed

            # if periods <= 0: ... # This check might be redundant now
            if abs(rate_monthly) <= FLOAT_ATOL: # Zero interest rate
                logger.debug(f"Year {year}: Zero rate amort.")
                # If rate is zero, pre-calculated payment should be LoanAmt/TotalPeriods
                # Principal paid is the payment, capped by balance
                principal_payment_yr = min(current_balance, monthly_payment * MONTHS_PER_YEAR)
                interest_payment_yr = 0.0
                ending_balance = max(0.0, current_balance - principal_payment_yr)
                logger.debug(f"Year {year} (Amort, Zero Rate): I={interest_payment_yr:.2f}, P={principal_payment_yr:.2f}, EndBal={ending_balance:.2f}")
            else: # Standard Amortization
                # REMOVED: monthly_pmt = npf.pmt(...) - Use passed-in monthly_payment
                monthly_pmt = monthly_payment # Use the constant payment passed in
                if not np.isfinite(monthly_pmt):
                    logger.error(f"Passed monthly_pmt is non-finite ({monthly_pmt}) in Year {year}.")
                    raise ValueError("Passed monthly_payment invalid.")

                balance_temp = current_balance
                interest_payment_yr = 0.0
                principal_payment_yr = 0.0
                for m in range(MONTHS_PER_YEAR):
                    if balance_temp <= FLOAT_ATOL: break
                    monthly_interest = balance_temp * rate_monthly
                    # Use the constant monthly payment
                    monthly_principal = monthly_pmt - monthly_interest
                    # Ensure principal doesn't exceed remaining balance
                    monthly_principal = max(0.0, min(monthly_principal, balance_temp))

                    if not np.isfinite(monthly_interest) or not np.isfinite(monthly_principal):
                         logger.error(f"NaN detected in Amort loop - Year {year}, Month {m + 1}.")
                         raise ValueError("NaN detected during monthly amortization calculation.")

                    interest_payment_yr += monthly_interest
                    principal_payment_yr += monthly_principal
                    balance_temp -= monthly_principal

                    if not np.isfinite(balance_temp):
                         logger.error(f"Balance became non-finite in Amort loop - Year {year}, Month {m + 1}.")
                         raise ValueError("Balance became non-finite during monthly amortization.")

                ending_balance = balance_temp
                logger.debug(f"Year {year} (Amort): Interest={interest_payment_yr:.2f}, Principal={principal_payment_yr:.2f}, EndBalance={ending_balance:.2f}")
        else:
            logger.error(f"Invalid Loan Type '{current_loan_type}' encountered in debt service calculation Year {year}.")
            raise ValueError(f"Invalid Loan Type '{current_loan_type}'.")
    except Exception as e:
        logger.error(f"Error during Interest/Principal calculation for Year {year}: {e}", exc_info=True)
        interest_payment_yr, principal_payment_yr, ending_balance = np.nan, np.nan, np.nan

    # --- Final Checks and Return ---
    final_ending_balance = max(0.0, ending_balance) if np.isfinite(ending_balance) else np.nan

    values_to_check = {
        "Interest": interest_payment_yr, "Principal": principal_payment_yr,
        "EndBalance": final_ending_balance, "EffRate": effective_rate,
    }
    # Only check vol_comp if it was supposed to be calculated
    if current_is_variable_rate: # Check original flag passed into function
         values_to_check["VolComp"] = current_volatile_sofr_comp_this_year

    non_finite_vars = {k: v for k, v in values_to_check.items() if not np.isfinite(v)}
    if non_finite_vars: # Check if the dictionary of non-finite vars is non-empty
        failed_vars_str = ", ".join([f"{k}={v}" for k, v in non_finite_vars.items()])
        actual_vol_comp_val = current_volatile_sofr_comp_this_year
        logger.error(f"Detected non-finite value(s) before final return from debt service - Year {year}: {failed_vars_str}. Actual VolComp={actual_vol_comp_val}. Returning NaNs.")
        return np.nan, np.nan, np.nan, np.nan, np.nan # Return failure state

    # If all checks pass, return the calculated values
    logger.info(f"Finished debt service Y{year}. Return: I={interest_payment_yr:.2f}, P={principal_payment_yr:.2f}, Bal={final_ending_balance:.2f}, Rate={effective_rate:.4f}, VolComp={current_volatile_sofr_comp_this_year}")
    return interest_payment_yr, principal_payment_yr, final_ending_balance, effective_rate, current_volatile_sofr_comp_this_year
