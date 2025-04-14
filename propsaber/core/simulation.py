"""
Contains the core simulation logic:
- run_monte_carlo: Orchestrates multiple simulation runs.
- run_single_simulation: Executes one full simulation path.
- Helper functions for state initialization, annual updates, exit calculations, IRR.

MODIFIED:
- _initialize_simulation_state: Calculates and stores initial monthly payment.
- _update_annual_state: Calculates new monthly payment on refi, passes
  stored monthly payment to calculate_debt_service.
- Refinancing logic confirmed as end-of-year.
"""

import numpy as np
import numpy_financial as npf
import pandas as pd
import random
import logging
import time
import copy
from typing import List, Dict, Any, Optional, Tuple
from joblib import Parallel, delayed
from .utils import simulation_error_handler

# Import necessary components from other core modules using relative imports
from .inputs import SimulationInputs
from .constants import (
    VAR_IDX, NUM_CORRELATED_VARS, FLOAT_ATOL, MONTHS_PER_YEAR,
    LOAN_TYPE_IO, LOAN_TYPE_AMORT
)
from .utils import (
    get_correlation_matrix, generate_correlated_shocks, get_valid_paths
)
from .debt import calculate_debt_service

logger = logging.getLogger(__name__)

# --- Simulation Helper Functions ---

@simulation_error_handler
def update_gbm_value(current_value: float, mean_growth: float, std_dev: float, shock: float) -> float:
    if current_value <= FLOAT_ATOL:
        return 0.0
    mu = mean_growth / 100.0
    sigma = max(0.0, std_dev / 100.0)
    drift = mu - 0.5 * sigma**2
    vol = sigma
    try:
        new_value = current_value * np.exp(drift + vol * shock)
        if not np.isfinite(new_value):
            logger.warning(f"GBM non-finite. Inputs: {current_value}, {mu}, {sigma}, {shock}. Result: {new_value}")
            return 0.0
        return max(0.0, new_value)
    except OverflowError:
        logger.warning(f"OverflowError in GBM. Inputs: {current_value}, {mu}, {sigma}, {shock}.")
        return 0.0
    except Exception as e:
        logger.error(f"Unexpected error in update_gbm_value: {e}", exc_info=True)
        return 0.0

@simulation_error_handler
def update_vacancy_ou(current_vacancy: float, target_vacancy: float, reversion_speed: float, volatility: float, shock: float) -> float:
    try:
        drift = reversion_speed * (target_vacancy - current_vacancy)
        vol = max(0.0, volatility)
        diffusion = vol * shock
        new_vacancy = current_vacancy + drift + diffusion
        if not np.isfinite(new_vacancy):
            logger.warning(f"OU Vacancy non-finite. Inputs: {current_vacancy}, {target_vacancy}, {reversion_speed}, {volatility}, {shock}. Result: {new_vacancy}")
            return max(0.0, min(1.0, target_vacancy))
        return max(0.0, min(1.0, new_vacancy))
    except Exception as e:
        logger.error(f"Unexpected error in update_vacancy_ou: {e}", exc_info=True)
        return max(0.0, min(1.0, target_vacancy))

@simulation_error_handler
def update_regime(current_regime: int, p_normal_to_recession: float, p_recession_to_normal: float) -> int:
    transition_prob = random.random()
    if current_regime == 0:
        return 1 if transition_prob < p_normal_to_recession else 0
    else:
        return 0 if transition_prob < p_recession_to_normal else 1

@simulation_error_handler
def _initialize_simulation_state(inputs: SimulationInputs) -> Optional[Dict[str, Any]]:
    """
    Initializes the state dictionary for a single simulation run.
    MODIFIED: Calculates and stores initial monthly payment.
    Returns None on critical error.
    """
    try:
        initial_deviation = inputs.market_rent_deviation_pct
        market_rent_per_unit = inputs.base_rent
        if not np.isclose(1.0 + initial_deviation, 0.0, atol=FLOAT_ATOL):
            normal_rent_per_unit = market_rent_per_unit / (1.0 + initial_deviation)
        else:
            logger.warning("Initial market rent deviation is -100%. Setting initial fair value rent equal to market rent.")
            normal_rent_per_unit = market_rent_per_unit

        initial_monthly_payment = 0.0
        loan_amount = inputs.loan_amount
        interest_rate = inputs.interest_rate
        loan_term_yrs = inputs.loan_term_yrs
        loan_type = inputs.loan_type
        is_variable = inputs.is_variable_rate

        if not is_variable and loan_type == LOAN_TYPE_AMORT:
            rate_monthly = interest_rate / MONTHS_PER_YEAR
            periods = loan_term_yrs * MONTHS_PER_YEAR
            if periods > 0 and abs(rate_monthly) > FLOAT_ATOL:
                try:
                    initial_monthly_payment = npf.pmt(rate_monthly, periods, -loan_amount)
                    if not np.isfinite(initial_monthly_payment):
                        logger.error(f"Initial npf.pmt failed (non-finite). Rate={rate_monthly}, Periods={periods}, PV={-loan_amount}")
                        initial_monthly_payment = 0.0
                except Exception as pmt_e:
                    logger.error(f"Error calculating initial npf.pmt: {pmt_e}", exc_info=True)
                    initial_monthly_payment = 0.0
            elif periods > 0:
                initial_monthly_payment = loan_amount / periods
            logger.info(f"Initialized with Monthly Payment: {initial_monthly_payment:.2f}")

        state = {
            "market_rent_per_unit": market_rent_per_unit,
            "normal_rent_per_unit": normal_rent_per_unit,
            "other_income": inputs.mean_other_income,
            "expenses": inputs.mean_expense,
            "vacancy_rate": inputs.current_vacancy,
            "loan_balance": loan_amount,
            "current_annual_capex": inputs.initial_capex,
            "prev_rent_shock": 0.0,
            "prev_other_inc_shock": 0.0,
            "prev_expense_shock": 0.0,
            "current_regime": 0,
            "prev_market_rent": market_rent_per_unit,
            "prev_other_income": inputs.mean_other_income,
            "prev_expenses": inputs.mean_expense,
            "prev_capex": inputs.initial_capex,
            "prev_volatile_sofr_component": None,
            "current_loan_amount": loan_amount,
            "current_loan_term_yrs": loan_term_yrs,
            "current_interest_rate": interest_rate,
            "current_loan_type": loan_type,
            "current_is_variable_rate": is_variable,
            "last_refi_year": -1,
            "current_monthly_payment": initial_monthly_payment,
            "years": list(range(1, inputs.hold_period + 1)),
            "annual_market_rent_per_unit": [],
            "annual_normal_rent_per_unit": [],
            "annual_potential_rent": [],
            "annual_vacancy_rate": [],
            "annual_vacancy_loss": [],
            "annual_effective_gross_rent": [],
            "annual_other_income": [],
            "annual_effective_gross_income": [],
            "annual_expenses": [],
            "annual_capex": [],
            "annual_noi": [],
            "annual_interest": [],
            "annual_principal": [],
            "annual_unlevered_cf": [],
            "annual_levered_cf": [],
            "annual_loan_balance": [],
            "annual_rent_growth_pct": [],
            "annual_other_income_growth_pct": [],
            "annual_expense_growth_pct": [],
            "annual_capex_growth_pct": [],
            "annual_interest_rates": [],
            "annual_refi_costs": [],
            "annual_refi_proceeds_net": [],
            "annual_property_value_estimate": [],
            "annual_ltv_estimate": [],
        }
        initial_numeric_state = [
            state['market_rent_per_unit'], state['normal_rent_per_unit'], state['other_income'],
            state['expenses'], state['vacancy_rate'], state['loan_balance'], state['current_annual_capex']
        ]
        if not all(np.isfinite(v) for v in initial_numeric_state):
            logger.error(f"Non-finite value detected in initial state: {state}")
            return None

        logger.debug(f"Initialized simulation state: {state}")
        return state
    except Exception as e:
        logger.error(f"Error during simulation state initialization: {e}", exc_info=True)
        return None

@simulation_error_handler
def _update_shocks(inputs: SimulationInputs, state: Dict[str, Any], raw_shock_row: np.ndarray) -> Optional[Dict[str, float]]:
    try:
        phi = max(0.0, min(0.999, inputs.growth_persistence_phi))
        sqrt_term = 1.0 - phi**2
        ar_factor = np.sqrt(sqrt_term) if sqrt_term >= 0 else 0.0

        current_rent_shock = phi * state["prev_rent_shock"] + ar_factor * raw_shock_row[VAR_IDX["Rent"]]
        current_other_inc_shock = phi * state["prev_other_inc_shock"] + ar_factor * raw_shock_row[VAR_IDX["OtherInc"]]
        current_expense_shock = phi * state["prev_expense_shock"] + ar_factor * raw_shock_row[VAR_IDX["Expense"]]
        vacancy_shock = raw_shock_row[VAR_IDX["Vacancy"]]
        capex_shock = current_expense_shock

        updated_shocks = {
            "rent_shock": current_rent_shock,
            "other_inc_shock": current_other_inc_shock,
            "expense_shock": current_expense_shock,
            "vacancy_shock": vacancy_shock,
            "capex_shock": capex_shock
        }
        if not all(np.isfinite(v) for v in updated_shocks.values()):
            logger.error(f"Non-finite shock: {updated_shocks}")
            return None
        logger.debug(f"Updated shocks (AR1 applied): {updated_shocks}")
        return updated_shocks
    except Exception as e:
        logger.error(f"Error updating shocks: {e}", exc_info=True)
        return None

@simulation_error_handler
def _update_annual_state(
    year_idx: int,
    year: int,
    inputs: SimulationInputs,
    state: Dict[str, Any],
    raw_shock_row: np.ndarray,
    forward_curve: Dict[int, float],
    std_dev_curve: Dict[int, float]
) -> Optional[Dict[str, Any]]:
    """
    Updates the simulation state for a single year.
    MODIFIED: Calculates new monthly payment on refi, passes
              stored monthly payment to calculate_debt_service.
    Refinancing logic confirmed as end-of-year.
    """
    try:
        refi_costs_this_year = 0.0
        refi_proceeds_net_this_year = 0.0
        property_value_estimate_this_year = np.nan
        ltv_estimate_this_year = np.nan
        refinance_occurred_this_year = False
        new_loan_details = {}

        shocks = _update_shocks(inputs, state, raw_shock_row)
        if shocks is None:
            logger.error(f"Year {year}: Failed to update shocks.")
            return None
        state["current_regime"] = update_regime(state["current_regime"], inputs.transition_normal_to_recession, inputs.transition_recession_to_normal)
        logger.debug(f"Year {year}: Regime={state['current_regime']}")
        if state["current_regime"] == 1:
            current_mean_rent_growth, current_rent_growth_stddev = inputs.recession_growth_mean, inputs.recession_growth_vol
        else:
            current_mean_rent_growth, current_rent_growth_stddev = inputs.normal_growth_mean, inputs.normal_growth_vol

        normal_rent_prev = state["normal_rent_per_unit"]
        state["normal_rent_per_unit"] = update_gbm_value(normal_rent_prev, current_mean_rent_growth, current_rent_growth_stddev, shocks["rent_shock"])
        convergence_yrs = max(1, inputs.market_convergence_years)
        current_deviation_factor = inputs.market_rent_deviation_pct * max(0.0, 1.0 - (year_idx + 1.0) / convergence_yrs)
        state["market_rent_per_unit"] = max(0.0, state["normal_rent_per_unit"] * (1.0 + current_deviation_factor))
        state["vacancy_rate"] = update_vacancy_ou(state["vacancy_rate"], inputs.stabilized_vacancy, inputs.vacancy_reversion_speed, inputs.vacancy_volatility, shocks["vacancy_shock"])
        state["other_income"] = update_gbm_value(state["other_income"], inputs.mean_other_income_growth, inputs.other_income_stddev, shocks["other_inc_shock"])
        state["expenses"] = update_gbm_value(state["expenses"], inputs.mean_expense_growth, inputs.expense_stddev, shocks["expense_shock"])
        state["current_annual_capex"] = update_gbm_value(state["current_annual_capex"], inputs.mean_capex_growth, inputs.capex_stddev, shocks["capex_shock"])
        capex_this_year = state["current_annual_capex"]
        current_values = [state["normal_rent_per_unit"], state["market_rent_per_unit"], state["vacancy_rate"], state["other_income"], state["expenses"], capex_this_year]
        if not all(np.isfinite(v) for v in current_values):
            logger.error(f"Year {year}: Non-finite core variable.")
            return None
        logger.debug(f"Year {year}: MarketRent={state['market_rent_per_unit']:.2f}, NormalRent={state['normal_rent_per_unit']:.2f}, Vacancy={state['vacancy_rate']:.3f}, OthInc={state['other_income']:.0f}, Exp={state['expenses']:.0f}, CapEx={capex_this_year:.0f}")

        rent_growth = ((state["market_rent_per_unit"] / state["prev_market_rent"] - 1.0) * 100.0) if state["prev_market_rent"] > FLOAT_ATOL else 0.0
        other_inc_growth = ((state["other_income"] / state["prev_other_income"] - 1.0) * 100.0) if state["prev_other_income"] > FLOAT_ATOL else 0.0
        expense_growth = ((state["expenses"] / state["prev_expenses"] - 1.0) * 100.0) if state["prev_expenses"] > FLOAT_ATOL else 0.0
        capex_growth = ((capex_this_year / state["prev_capex"] - 1.0) * 100.0) if state["prev_capex"] > FLOAT_ATOL else 0.0

        potential_gross_rent = inputs.num_units * state["market_rent_per_unit"] * MONTHS_PER_YEAR
        vacancy_loss = potential_gross_rent * state["vacancy_rate"]
        effective_gross_rent = potential_gross_rent - vacancy_loss
        effective_gross_income = effective_gross_rent + state["other_income"]
        noi = effective_gross_income - state["expenses"]
        unlevered_cf = noi - capex_this_year
        logger.debug(f"Year {year}: PGR={potential_gross_rent:.0f}, VacLoss={vacancy_loss:.0f}, EGI={effective_gross_income:.0f}, NOI={noi:.0f}, UnlevCF={unlevered_cf:.0f}")
        if not all(np.isfinite(v) for v in [noi, unlevered_cf]):
            logger.error(f"Year {year}: Non-finite NOI or Unlev CF.")
            return None

        current_monthly_payment = state.get('current_monthly_payment', 0.0)

        interest_yr, principal_yr, ending_loan_balance_before_refi, effective_rate, current_volatile_sofr_comp = calculate_debt_service(
            current_loan_type=state['current_loan_type'],
            current_interest_rate=state['current_interest_rate'],
            current_is_variable_rate=state['current_is_variable_rate'],
            current_balance=state["loan_balance"],
            monthly_payment=current_monthly_payment,
            year=year,
            sofr_spread=inputs.sofr_spread,
            forward_curve=forward_curve,
            std_dev_curve=std_dev_curve,
            sofr_floor=inputs.sofr_floor,
            rate_persistence_phi=inputs.rate_persistence_phi,
            volatility_scalar=inputs.volatility_scalar,
            prev_volatile_sofr_comp=state["prev_volatile_sofr_component"]
        )
        logger.debug(f"Year {year}: DebtService Results (Pre-Refi Terms): I={interest_yr}, P={principal_yr}, EndBal={ending_loan_balance_before_refi}, EffRate={effective_rate}, VolComp={current_volatile_sofr_comp}")

        was_variable_rate = state['current_is_variable_rate']
        results_to_check = [interest_yr, principal_yr, ending_loan_balance_before_refi, effective_rate]
        if was_variable_rate:
            results_to_check.append(current_volatile_sofr_comp)

        if any(v is None or not np.isfinite(v) for v in results_to_check):
            logger.error(f"Year {year}: Debt service failed. Results: I={interest_yr}, P={principal_yr}, Bal={ending_loan_balance_before_refi}, Rate={effective_rate}, VolComp={current_volatile_sofr_comp}.")
            return None

        if inputs.enable_refinancing and year == inputs.refi_year and state["last_refi_year"] < year:
            logger.info(f"Year {year}: Performing End-of-Year Refinance Calculation.")
            refinance_occurred_this_year = True

            refi_cap_rate = inputs.mean_exit_cap_rate / 100.0
            if noi <= 0 or refi_cap_rate <= FLOAT_ATOL:
                logger.warning(f"Year {year}: Cannot estimate property value for refinance (NOI={noi:.0f}, CapRate={refi_cap_rate:.4f}). Skipping refi.")
                property_value_estimate_this_year = 0.0
                refinance_occurred_this_year = False
            else:
                property_value_estimate_this_year = noi / refi_cap_rate
                logger.debug(f"Year {year} Refi: Est. Value = {property_value_estimate_this_year:,.0f}")

                new_loan_amount = property_value_estimate_this_year * inputs.refi_new_ltv
                logger.debug(f"Year {year} Refi: New Loan Amt = {new_loan_amount:,.0f}")

                refi_costs_this_year = new_loan_amount * inputs.refi_costs_pct_loan
                logger.debug(f"Year {year} Refi: Costs = {refi_costs_this_year:,.0f}")

                old_loan_balance_start_year = state['loan_balance']
                cash_out_in = new_loan_amount - old_loan_balance_start_year
                refi_proceeds_net_this_year = cash_out_in - refi_costs_this_year
                logger.debug(f"Year {year} Refi: Net Proceeds = {refi_proceeds_net_this_year:,.0f}")

                max_curve_year = max(forward_curve.keys()) if forward_curve else 1
                base_sofr_refi_yr = forward_curve.get(year, forward_curve.get(max_curve_year, 0.0))
                new_fixed_rate = base_sofr_refi_yr + (inputs.refi_fixed_rate_spread_to_sofr / 100.0)
                logger.debug(f"Year {year} Refi: New Fixed Rate = {new_fixed_rate:.4f}")

                new_monthly_payment = 0.0
                new_rate_monthly = new_fixed_rate / MONTHS_PER_YEAR
                new_periods = inputs.refi_new_amort_period * MONTHS_PER_YEAR
                if new_periods > 0 and new_loan_amount > FLOAT_ATOL:
                    if abs(new_rate_monthly) > FLOAT_ATOL:
                        try:
                            new_monthly_payment = npf.pmt(new_rate_monthly, new_periods, -new_loan_amount)
                            if not np.isfinite(new_monthly_payment):
                                new_monthly_payment = 0.0
                        except Exception as pmt_e:
                            logger.error(f"Error calculating refi pmt: {pmt_e}")
                            new_monthly_payment = 0.0
                    else:
                        new_monthly_payment = new_loan_amount / new_periods
                logger.info(f"Year {year} Refi: Calculated New Monthly Payment: {new_monthly_payment:.2f}")

                new_loan_details = {
                    'loan_balance': new_loan_amount,
                    'current_loan_amount': new_loan_amount,
                    'current_loan_term_yrs': inputs.refi_new_amort_period,
                    'current_interest_rate': new_fixed_rate,
                    'current_is_variable_rate': False,
                    'current_loan_type': LOAN_TYPE_AMORT,
                    'last_refi_year': year,
                    'prev_volatile_sofr_component': np.nan,
                    'current_monthly_payment': new_monthly_payment
                }

                if property_value_estimate_this_year > FLOAT_ATOL and np.isfinite(new_loan_amount):
                    ltv_estimate_this_year = new_loan_amount / property_value_estimate_this_year
                else:
                    ltv_estimate_this_year = np.nan
        else:
            refi_cap_rate = inputs.mean_exit_cap_rate / 100.0
            if noi > 0 and refi_cap_rate > FLOAT_ATOL:
                property_value_estimate_this_year = noi / refi_cap_rate
                if np.isfinite(ending_loan_balance_before_refi):
                    ltv_estimate_this_year = ending_loan_balance_before_refi / property_value_estimate_this_year
            else:
                property_value_estimate_this_year = 0.0
                ltv_estimate_this_year = np.nan

        if np.isfinite(unlevered_cf) and np.isfinite(interest_yr) and np.isfinite(principal_yr):
            levered_cf = unlevered_cf - interest_yr - principal_yr + refi_proceeds_net_this_year
        else:
            levered_cf = np.nan
        logger.debug(f"Year {year}: LevCF={levered_cf:.0f} (includes Refi Proceeds/Costs: {refi_proceeds_net_this_year:.0f})")
        if not np.isfinite(levered_cf):
            logger.error(f"Year {year}: Levered CF non-finite.")
            return None

        state["annual_market_rent_per_unit"].append(state["market_rent_per_unit"])
        state["annual_normal_rent_per_unit"].append(state["normal_rent_per_unit"])
        state["annual_potential_rent"].append(potential_gross_rent)
        state["annual_vacancy_rate"].append(state["vacancy_rate"])
        state["annual_vacancy_loss"].append(vacancy_loss)
        state["annual_effective_gross_rent"].append(effective_gross_rent)
        state["annual_other_income"].append(state["other_income"])
        state["annual_effective_gross_income"].append(effective_gross_income)
        state["annual_expenses"].append(state["expenses"])
        state["annual_capex"].append(capex_this_year)
        state["annual_noi"].append(noi)
        state["annual_interest"].append(interest_yr)
        state["annual_principal"].append(principal_yr)
        state["annual_unlevered_cf"].append(unlevered_cf)
        state["annual_levered_cf"].append(levered_cf)
        state["annual_loan_balance"].append(ending_loan_balance_before_refi)
        state["annual_rent_growth_pct"].append(rent_growth)
        state["annual_other_income_growth_pct"].append(other_inc_growth)
        state["annual_expense_growth_pct"].append(expense_growth)
        state["annual_capex_growth_pct"].append(capex_growth)
        state["annual_interest_rates"].append(effective_rate)
        state["annual_refi_costs"].append(refi_costs_this_year)
        state["annual_refi_proceeds_net"].append(refi_proceeds_net_this_year)
        state["annual_property_value_estimate"].append(property_value_estimate_this_year)
        state["annual_ltv_estimate"].append(ltv_estimate_this_year)

        if refinance_occurred_this_year and new_loan_details:
            logger.info(f"Year {year}: Applying end-of-year refinance terms for start of Year {year + 1}.")
            state.update(new_loan_details)
        else:
            state['loan_balance'] = ending_loan_balance_before_refi
            if state['current_loan_type'] == LOAN_TYPE_AMORT and state['current_loan_term_yrs'] > 0:
                state['current_loan_term_yrs'] -= 1
            state["prev_volatile_sofr_component"] = current_volatile_sofr_comp
            state['current_monthly_payment'] = state.get('current_monthly_payment', 0.0)

        state["prev_market_rent"] = state["market_rent_per_unit"]
        state["prev_other_income"] = state["other_income"]
        state["prev_expenses"] = state["expenses"]
        state["prev_capex"] = capex_this_year
        state["prev_rent_shock"] = shocks["rent_shock"]
        state["prev_other_inc_shock"] = shocks["other_inc_shock"]
        state["prev_expense_shock"] = shocks["expense_shock"]

    except Exception as e:
        logger.error(f"Unhandled exception in _update_annual_state Year {year}: {e}", exc_info=True)
        return None

    return state

def _calculate_exit_values(inputs: SimulationInputs, state: Dict[str, Any], L_matrix: np.ndarray) -> Tuple[float, float, float, float, float]:
    try:
        terminal_year_rent_growth_pct = state["annual_rent_growth_pct"][-1] if state.get("annual_rent_growth_pct") and np.isfinite(state["annual_rent_growth_pct"][-1]) else inputs.normal_growth_mean
        if not np.isfinite(terminal_year_rent_growth_pct):
            logger.warning("Terminal rent growth non-finite")
            terminal_year_rent_growth_pct = inputs.normal_growth_mean
        rent_growth_surprise_pct = terminal_year_rent_growth_pct - inputs.normal_growth_mean
        rent_growth_cap_adj_pct = inputs.exit_cap_rent_growth_sensitivity * rent_growth_surprise_pct
        exit_shocks = generate_correlated_shocks(L_matrix, 1)
        if exit_shocks is None or exit_shocks.shape[1] <= VAR_IDX["ExitCap"]:
            logger.error("Failed exit shock gen.")
            exit_cap_shock = 0.0
        else:
            exit_cap_shock = exit_shocks[0, VAR_IDX["ExitCap"]]
        base_cap_deviation_pct = exit_cap_shock * inputs.exit_cap_rate_stddev
        sim_exit_cap_pct = inputs.mean_exit_cap_rate + base_cap_deviation_pct + rent_growth_cap_adj_pct
        sim_exit_cap_pct_clamped = max(1.0, sim_exit_cap_pct)
        sim_exit_cap = sim_exit_cap_pct_clamped / 100.0
        logger.debug(f"Exit Calc: TermRentGrowth={terminal_year_rent_growth_pct:.1f}%, SimCap={sim_exit_cap_pct_clamped:.2f}%")
        exit_noi = state["annual_noi"][-1] if state.get("annual_noi") and np.isfinite(state["annual_noi"][-1]) else 0.0
        if exit_noi <= 0:
            logger.warning(f"Terminal NOI <= 0 ({exit_noi:.0f}).")
            last_value_est = state.get("annual_property_value_estimate", [0])[-1]
            terminal_value_gross = max(inputs.exit_floor_value, last_value_est if np.isfinite(last_value_est) else 0.0)
        elif sim_exit_cap <= FLOAT_ATOL:
            logger.warning(f"Sim exit cap near zero ({sim_exit_cap:.4f}).")
            last_value_est = state.get("annual_property_value_estimate", [0])[-1]
            terminal_value_gross = max(inputs.exit_floor_value, last_value_est if np.isfinite(last_value_est) else 0.0)
        else:
            terminal_value_gross = exit_noi / sim_exit_cap
        terminal_value_gross = max(terminal_value_gross, inputs.exit_floor_value)
        transaction_costs = terminal_value_gross * inputs.transaction_cost_pct
        terminal_value_net = max(terminal_value_gross - transaction_costs, 0.0)
        logger.debug(f"Exit Calc: ExitNOI={exit_noi:.0f}, SimCap={sim_exit_cap:.4f}, GrossValue={terminal_value_gross:.0f}, NetValue={terminal_value_net:.0f}")
        last_unlevered_cf = state["annual_unlevered_cf"][-1] if state.get("annual_unlevered_cf") and np.isfinite(state["annual_unlevered_cf"][-1]) else 0.0
        last_levered_cf = state["annual_levered_cf"][-1] if state.get("annual_levered_cf") and np.isfinite(state["annual_levered_cf"][-1]) else 0.0
        loan_payoff = state["annual_loan_balance"][-1] if state.get("annual_loan_balance") and np.isfinite(state["annual_loan_balance"][-1]) else 0.0
        logger.debug(f"Exit Calc: LoanPayoff={loan_payoff:.0f}")
        final_unlevered_cf = last_unlevered_cf + terminal_value_net
        final_levered_cf = last_levered_cf + terminal_value_net - loan_payoff
        logger.debug(f"Exit Calc: FinalUnlevCF={final_unlevered_cf:.0f}, FinalLevCF={final_levered_cf:.0f}")
        results_tuple = (sim_exit_cap, terminal_value_net, final_unlevered_cf, final_levered_cf, terminal_year_rent_growth_pct)
        if not all(np.isfinite(v) for v in results_tuple):
            logger.error(f"Non-finite exit results: {results_tuple}")
            return np.nan, np.nan, np.nan, np.nan, np.nan
        return results_tuple
    except Exception as e:
        logger.error(f"Error during exit value calc: {e}", exc_info=True)
        return np.nan, np.nan, np.nan, np.nan, np.nan

def _calculate_irr(inputs: SimulationInputs, state: Dict[str, Any], final_unlevered_cf: float, final_levered_cf: float) -> Tuple[float, float, List[float], List[float], bool, bool]:
    initial_investment = -inputs.purchase_price
    annual_unlevered_cfs = state.get("annual_unlevered_cf", [])
    final_unlevered_cf_safe = final_unlevered_cf if np.isfinite(final_unlevered_cf) else 0.0
    if annual_unlevered_cfs and len(annual_unlevered_cfs) == inputs.hold_period:
        unlevered_op_cfs = [cf if np.isfinite(cf) else 0.0 for cf in annual_unlevered_cfs[:-1]]
        unlevered_stream = [initial_investment] + unlevered_op_cfs + [final_unlevered_cf_safe]
    else:
        logger.warning(f"Unlevered CF list mismatch/empty.")
        unlevered_stream = [initial_investment, final_unlevered_cf_safe]
    initial_equity_investment = -inputs.initial_equity
    annual_levered_cfs = state.get("annual_levered_cf", [])
    final_levered_cf_safe = final_levered_cf if np.isfinite(final_levered_cf) else 0.0
    if annual_levered_cfs and len(annual_levered_cfs) == inputs.hold_period:
        levered_op_cfs = [cf if np.isfinite(cf) else 0.0 for cf in annual_levered_cfs[:-1]]
        levered_stream = [initial_equity_investment] + levered_op_cfs + [final_levered_cf_safe]
    else:
        logger.warning(f"Levered CF list mismatch/empty.")
        levered_stream = [initial_equity_investment, final_levered_cf_safe]
    failed_unlevered = False
    failed_levered = False
    unlevered_irr = np.nan
    levered_irr = np.nan
    if not all(np.isfinite(cf) for cf in unlevered_stream):
        logger.warning(f"Non-finite unlevered stream.")
        failed_unlevered = True
    elif abs(initial_investment) <= FLOAT_ATOL:
        logger.warning("Zero initial investment.")
        failed_unlevered = True
    else:
        try:
            unlevered_irr = npf.irr(unlevered_stream)
        except ValueError as e:
            logger.debug(f"Unlevered IRR ValueError: {e}")
            failed_unlevered = True
        if not np.isfinite(unlevered_irr):
            failed_unlevered = True
            unlevered_irr = np.nan
    if not all(np.isfinite(cf) for cf in levered_stream):
        logger.warning(f"Non-finite levered stream.")
        failed_levered = True
    elif abs(initial_equity_investment) <= FLOAT_ATOL:
        logger.warning("Zero initial equity.")
        failed_levered = True
    else:
        try:
            levered_irr = npf.irr(levered_stream)
        except ValueError as e:
            logger.debug(f"Levered IRR ValueError: {e}")
            failed_levered = True
        if not np.isfinite(levered_irr):
            failed_levered = True
            levered_irr = np.nan
    logger.debug(f"IRR Calc Results: UnlevIRR={unlevered_irr}, LevIRR={levered_irr}, UnlevFail={failed_unlevered}, LevFail={failed_levered}")
    return unlevered_irr, levered_irr, unlevered_stream, levered_stream, failed_unlevered, failed_levered

@simulation_error_handler
def run_single_simulation(sim_index: int, inputs: SimulationInputs, L_matrix: np.ndarray, forward_curve: Dict[int, float], std_dev_curve: Dict[int, float]) -> Optional[Dict[str, Any]]:
    logger.info(f"Starting simulation run #{sim_index + 1}")
    state = None
    sim_exit_cap, exit_value_net, final_unlevered_cf, final_levered_cf = [np.nan] * 4
    unlevered_irr, levered_irr = np.nan, np.nan
    unlevered_stream, levered_stream = [], []
    failed_unlevered_irr, failed_levered_irr = True, True
    term_rent_growth = np.nan
    try:
        state = _initialize_simulation_state(inputs)
        if state is None:
            logger.error(f"Run #{sim_index + 1}: State init failed.")
            return None
        hold_period = max(1, inputs.hold_period)
        raw_shocks = generate_correlated_shocks(L_matrix, hold_period)
        if raw_shocks is None or raw_shocks.shape[0] != hold_period:
            logger.error(f"Run #{sim_index + 1}: Shock gen failed.")
            return None
        years_to_simulate = state.get("years", list(range(1, hold_period + 1)))
        for year_idx, year in enumerate(years_to_simulate):
            state = _update_annual_state(year_idx, year, inputs, state, raw_shocks[year_idx, :], forward_curve, std_dev_curve)
            if state is None:
                logger.error(f"Run #{sim_index + 1}: Failed update Year {year}.")
                return None
        if state is None:
            logger.error(f"Run #{sim_index + 1}: State None after loop.")
            return None
        sim_exit_cap, exit_value_net, final_unlevered_cf, final_levered_cf, term_rent_growth = _calculate_exit_values(inputs, state, L_matrix)
        if not all(np.isfinite(v) for v in [sim_exit_cap, exit_value_net, final_unlevered_cf, final_levered_cf, term_rent_growth]):
            logger.error(f"Run #{sim_index + 1}: Exit calc failed.")
        unlevered_irr, levered_irr, unlevered_stream, levered_stream, failed_unlevered_irr, failed_levered_irr = _calculate_irr(inputs, state, final_unlevered_cf, final_levered_cf)
        logger.debug(f"Run #{sim_index + 1}: IRR Results - Unlev={unlevered_irr}, Lev={levered_irr}")
    except Exception as e:
        logger.error(f"Error during single sim run #{sim_index + 1}: {e}", exc_info=True)
        return None
    expected_len = inputs.hold_period
    def _pad_list_safe(data_list: Optional[List[Any]], expected_len: int, pad_value: Any = np.nan) -> List[Any]:
        if data_list is None:
            return [pad_value] * expected_len
        base_list = list(data_list)
        actual_len = len(base_list)
        if actual_len == expected_len:
            return base_list
        elif actual_len < expected_len:
            return base_list + ([pad_value] * (expected_len - actual_len))
        else:
            return base_list[:expected_len]
    if state is None:
        logger.error(f"Run #{sim_index + 1}: State None before results packaging.")
        return None
    results = {
        "unlevered_irr": unlevered_irr,
        "levered_irr": levered_irr,
        "exit_value_net": exit_value_net,
        "sim_exit_cap_rate": sim_exit_cap,
        "years": _pad_list_safe(state.get("years"), expected_len, 0),
        "rent_per_unit": _pad_list_safe(state.get("annual_market_rent_per_unit"), expected_len),
        "normal_rent_per_unit": _pad_list_safe(state.get("annual_normal_rent_per_unit"), expected_len),
        "potential_rent": _pad_list_safe(state.get("annual_potential_rent"), expected_len),
        "vacancy_rate": _pad_list_safe(state.get("annual_vacancy_rate"), expected_len),
        "vacancy_loss": _pad_list_safe(state.get("annual_vacancy_loss"), expected_len),
        "egr": _pad_list_safe(state.get("annual_effective_gross_rent"), expected_len),
        "other_income": _pad_list_safe(state.get("annual_other_income"), expected_len),
        "egi": _pad_list_safe(state.get("annual_effective_gross_income"), expected_len),
        "expenses": _pad_list_safe(state.get("annual_expenses"), expected_len),
        "capex": _pad_list_safe(state.get("annual_capex"), expected_len),
        "noi": _pad_list_safe(state.get("annual_noi"), expected_len),
        "interest": _pad_list_safe(state.get("annual_interest"), expected_len),
        "principal": _pad_list_safe(state.get("annual_principal"), expected_len),
        "unlevered_cf": _pad_list_safe(state.get("annual_unlevered_cf"), expected_len),
        "levered_cf": _pad_list_safe(state.get("annual_levered_cf"), expected_len),
        "loan_balance": _pad_list_safe(state.get("annual_loan_balance"), expected_len),
        "unlevered_stream_irr": unlevered_stream,
        "levered_stream_irr": levered_stream,
        "terminal_yr_rent_growth_pct": term_rent_growth,
        "rent_growth_pct": _pad_list_safe(state.get("annual_rent_growth_pct"), expected_len),
        "other_inc_growth_pct": _pad_list_safe(state.get("annual_other_income_growth_pct"), expected_len),
        "expense_growth_pct": _pad_list_safe(state.get("annual_expense_growth_pct"), expected_len),
        "capex_growth_pct": _pad_list_safe(state.get("annual_capex_growth_pct"), expected_len),
        "interest_rates": _pad_list_safe(state.get("annual_interest_rates"), expected_len),
        "refi_costs": _pad_list_safe(state.get("annual_refi_costs"), expected_len),
        "refi_proceeds_net": _pad_list_safe(state.get("annual_refi_proceeds_net"), expected_len),
        "property_value_estimate": _pad_list_safe(state.get("annual_property_value_estimate"), expected_len),
        "ltv_estimate": _pad_list_safe(state.get("annual_ltv_estimate"), expected_len),
        "failed_unlevered_irr": failed_unlevered_irr,
        "failed_levered_irr": failed_levered_irr
    }
    logger.info(f"Finished simulation run #{sim_index + 1} successfully.")
    return results

@simulation_error_handler
def run_monte_carlo(inputs: SimulationInputs, num_simulations: int, forward_curve: Dict[int, float], std_dev_curve: Dict[int, float]) -> Dict[str, Any]:
    start_time = time.time()
    logger.info(f"Starting Monte Carlo: {num_simulations} sims, Hold: {inputs.hold_period} yrs.")
    try:
        corr_matrix = get_correlation_matrix(inputs)
        L_matrix = np.linalg.cholesky(corr_matrix)
        logger.info("Computed Cholesky.")
    except np.linalg.LinAlgError:
        logger.warning("Cholesky failed. Using identity.")
        L_matrix = np.identity(NUM_CORRELATED_VARS)
    except Exception as e:
        logger.error(f"Pre-computation error: {e}", exc_info=True)
        return {"error": f"Pre-computation failed: {e}"}
    all_results = []
    try:
        logger.info(f"Starting parallel execution: {num_simulations} tasks...")
        with Parallel(n_jobs=-1, backend="loky") as parallel:
            all_results = parallel(
                delayed(run_single_simulation)(i, inputs, L_matrix, forward_curve, std_dev_curve)
                for i in range(num_simulations)
            )
        logger.info(f"Parallel exec completed. Received {len(all_results)} results.")
    except Exception as e:
        logger.error(f"Parallel exec error: {e}", exc_info=True)
        return {"error": f"Simulation parallel exec failed: {e}"}
    sim_results_completed = [r for r in all_results if r is not None and isinstance(r, dict)]
    num_total_sims = len(all_results)
    num_completed_sims = len(sim_results_completed)
    if num_completed_sims < num_total_sims:
        logger.warning(f"Filtered out {num_total_sims - num_completed_sims} failed runs.")
    if not sim_results_completed:
        logger.error("No simulations completed successfully.")
        return {"error": "No simulations completed.", "inputs_used": inputs.to_dict()}
    metrics: Dict[str, Any] = {}
    risk_metrics: Dict[str, Any] = {}
    plot_data: Dict[str, Any] = {}
    raw_metrics_lists = {
        "unlevered_irrs": [],
        "levered_irrs": [],
        "exit_values": [],
        "exit_caps": [],
        "loan_payoffs": []
    }
    unlevered_failure_count = 0
    levered_failure_count = 0
    for r in sim_results_completed:
        raw_metrics_lists["unlevered_irrs"].append(r.get("unlevered_irr", np.nan))
        raw_metrics_lists["levered_irrs"].append(r.get("levered_irr", np.nan))
        raw_metrics_lists["exit_values"].append(r.get("exit_value_net", np.nan))
        raw_metrics_lists["exit_caps"].append(r.get("sim_exit_cap_rate", np.nan))
        lb = r.get("loan_balance")
        raw_metrics_lists["loan_payoffs"].append(lb[-1] if isinstance(lb, list) and lb and np.isfinite(lb[-1]) else np.nan)
        if r.get("failed_unlevered_irr", True):
            unlevered_failure_count += 1
        if r.get("failed_levered_irr", True):
            levered_failure_count += 1
    def calculate_finite_stats(data: List[float], prefix: str) -> Tuple[Dict[str, Any], List[float]]:
        finite_data = [x for x in data if x is not None and np.isfinite(x)]
        stats = {}
        if finite_data:
            stats[f"mean_{prefix}"] = float(np.mean(finite_data))
            stats[f"p05_{prefix}"] = float(np.percentile(finite_data, 5))
            stats[f"p95_{prefix}"] = float(np.percentile(finite_data, 95))
            stats[f"median_{prefix}"] = float(np.median(finite_data))
        else:
            logger.warning(f"No finite data for '{prefix}'.")
            stats[f"mean_{prefix}"] = np.nan
            stats[f"p05_{prefix}"] = np.nan
            stats[f"p95_{prefix}"] = np.nan
            stats[f"median_{prefix}"] = np.nan
        return stats, finite_data
    unlevered_irr_stats, finite_unlevered = calculate_finite_stats(raw_metrics_lists["unlevered_irrs"], "unlevered_irr")
    metrics.update(unlevered_irr_stats)
    levered_irr_stats, finite_levered = calculate_finite_stats(raw_metrics_lists["levered_irrs"], "levered_irr")
    metrics.update(levered_irr_stats)
    exit_value_stats, finite_exit_values = calculate_finite_stats(raw_metrics_lists["exit_values"], "exit_value")
    metrics.update(exit_value_stats)
    finite_exit_caps = [cap for cap in raw_metrics_lists["exit_caps"] if cap is not None and np.isfinite(cap)]
    metrics["mean_exit_cap"] = float(np.mean(finite_exit_caps)) if finite_exit_caps else np.nan
    metrics["median_exit_cap"] = float(np.median(finite_exit_caps)) if finite_exit_caps else np.nan
    finite_loan_payoffs = [p for p in raw_metrics_lists["loan_payoffs"] if p is not None and np.isfinite(p)]
    metrics["mean_loan_payoff"] = float(np.mean(finite_loan_payoffs)) if finite_loan_payoffs else np.nan
    risk_metric_keys = ["Std Dev IRR", "Sharpe Ratio", "Coefficient of Variation", "Prob. Loss (IRR < 0%)", "Prob. Below Hurdle", "Value at Risk (VaR 95%)", "Cond. VaR (CVaR 95%)"]
    if finite_levered:
        mean_l_irr = metrics.get("mean_levered_irr", np.nan)
        p05_l_irr = metrics.get("p05_levered_irr", np.nan)
        if np.isfinite(mean_l_irr):
            std_l_irr = np.std(finite_levered)
            risk_free = inputs.risk_free_rate
            hurdle = inputs.hurdle_rate
            risk_metrics["Std Dev IRR"] = std_l_irr
            risk_metrics["Sharpe Ratio"] = (mean_l_irr - risk_free) / std_l_irr if std_l_irr > FLOAT_ATOL else np.nan
            risk_metrics["Coefficient of Variation"] = std_l_irr / abs(mean_l_irr) if abs(mean_l_irr) > FLOAT_ATOL else np.nan
            risk_metrics["Prob. Loss (IRR < 0%)"] = np.mean(np.array(finite_levered) < 0.0)
            risk_metrics["Prob. Below Hurdle"] = np.mean(np.array(finite_levered) < hurdle)
            risk_metrics["Value at Risk (VaR 95%)"] = p05_l_irr
            if np.isfinite(p05_l_irr):
                cvar_irrs = [irr for irr in finite_levered if irr <= p05_l_irr]
                risk_metrics["Cond. VaR (CVaR 95%)"] = np.mean(cvar_irrs) if cvar_irrs else np.nan
            else:
                risk_metrics["Cond. VaR (CVaR 95%)"] = np.nan
        else:
            logger.warning("Mean Levered IRR NaN, risk metrics skipped.")
            risk_metrics = {k: np.nan for k in risk_metric_keys}
    else:
        logger.warning("No finite Levered IRR data for risk metrics.")
        risk_metrics = {k: np.nan for k in risk_metric_keys}
    hold_period_actual = inputs.hold_period
    years_list = list(range(1, hold_period_actual + 1))
    plot_data["years"] = years_list
    rent_norm_plot = {}
    market_rent_paths = get_valid_paths(sim_results_completed, "rent_per_unit", hold_period_actual)
    normal_rent_paths = get_valid_paths(sim_results_completed, "normal_rent_per_unit", hold_period_actual)
    if market_rent_paths and normal_rent_paths:
        try:
            market_rent_arr = np.array(market_rent_paths)
            normal_rent_arr = np.array(normal_rent_paths)
            rent_norm_plot["market_p05"] = np.percentile(market_rent_arr, 5, axis=0).tolist()
            rent_norm_plot["market_p50"] = np.percentile(market_rent_arr, 50, axis=0).tolist()
            rent_norm_plot["market_p95"] = np.percentile(market_rent_arr, 95, axis=0).tolist()
            rent_norm_plot["normal_p50"] = np.percentile(normal_rent_arr, 50, axis=0).tolist()
        except Exception as e:
            logger.error(f"Error processing rent percentiles: {e}")
            rent_norm_plot = {}
    plot_data["rent_norm_plot"] = rent_norm_plot
    vacancy_plot_df = None
    vacancy_paths = get_valid_paths(sim_results_completed, "vacancy_rate", hold_period_actual)
    if vacancy_paths:
        try:
            vacancy_array = np.array(vacancy_paths)
            vacancy_df_prep = []
            for i, year in enumerate(years_list):
                for sim_run_vacancy in vacancy_array[:, i]:
                    vacancy_df_prep.append({"Year": year, "Vacancy Rate": sim_run_vacancy})
            vacancy_plot_df = pd.DataFrame(vacancy_df_prep)
        except Exception as e:
            logger.error(f"Error processing vacancy plot data: {e}")
            vacancy_plot_df = None
    plot_data["vacancy_plot_df"] = vacancy_plot_df
    scatter_plot_data = {"term_rent_growth_pct": [], "exit_cap_rate_pct": []}
    for r in sim_results_completed:
        term_rent = r.get("terminal_yr_rent_growth_pct")
        exit_cap = r.get("sim_exit_cap_rate")
        if term_rent is not None and exit_cap is not None:
            term_rent_safe = term_rent if np.isfinite(term_rent) else inputs.normal_growth_mean
            exit_cap_safe = exit_cap if np.isfinite(exit_cap) else inputs.mean_exit_cap_rate / 100.0
            scatter_plot_data["term_rent_growth_pct"].append(term_rent_safe)
            scatter_plot_data["exit_cap_rate_pct"].append(exit_cap_safe * 100.0)
        else:
            logger.warning(f"Missing values in sim: term_rent_growth_pct={term_rent}, sim_exit_cap_rate={exit_cap}")
    logger.info(f"Scatter plot data: term_rent_growth_pct={scatter_plot_data['term_rent_growth_pct']}, exit_cap_rate_pct={scatter_plot_data['exit_cap_rate_pct']}")
    plot_data["scatter_plot"] = scatter_plot_data
    avg_data = {}
    avg_data_keys = [
        "potential_rent", "vacancy_loss", "egr", "other_income", "egi", "expenses",
        "capex", "noi", "interest", "principal", "unlevered_cf", "levered_cf",
        "loan_balance", "vacancy_rate", "rent_per_unit", "rent_growth_pct",
        "other_inc_growth_pct", "expense_growth_pct", "capex_growth_pct",
        "interest_rates", "refi_costs", "refi_proceeds_net",
        "property_value_estimate", "ltv_estimate"
    ]
    for key in avg_data_keys:
        key_paths_finite = get_valid_paths(sim_results_completed, key, hold_period_actual)
        if key_paths_finite:
            try:
                avg_data[key] = np.mean(np.array(key_paths_finite), axis=0).tolist()
            except Exception as e:
                logger.error(f"Error averaging paths for key '{key}': {e}", exc_info=True)
                avg_data[key] = [np.nan] * hold_period_actual
        else:
            logger.warning(f"No valid paths found for averaging key '{key}'.")
            avg_data[key] = [np.nan] * hold_period_actual
    plot_data["avg_cash_flows"] = avg_data
    final_metrics = {k: float(v) if isinstance(v, (np.number, np.bool_)) else v for k, v in metrics.items()}
    final_risk_metrics = {k: float(v) if isinstance(v, (np.number, np.bool_)) else v for k, v in risk_metrics.items()}
    end_time = time.time()
    logger.info(f"Monte Carlo finished. Completed: {num_completed_sims}/{num_total_sims}. Time: {end_time - start_time:.2f}s.")
    return {
        "metrics": final_metrics,
        "risk_metrics": final_risk_metrics,
        "plot_data": plot_data,
        "raw_results_for_audit": sim_results_completed,
        "finite_unlevered_irrs": finite_unlevered,
        "finite_levered_irrs": finite_levered,
        "finite_exit_values": finite_exit_values,
        "finite_exit_caps": finite_exit_caps,
        "unlev_forced_failures": unlevered_failure_count,
        "lev_forced_failures": levered_failure_count,
        "config_num_simulations": num_simulations,
        "config_hold_period": inputs.hold_period,
        "num_simulations_completed": num_completed_sims,
    }
