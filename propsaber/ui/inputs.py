# propsaber/ui/inputs.py
"""
Contains functions for rendering Streamlit input widgets in the sidebar
and aggregating their values. Extracted and adapted from the main script.

MODIFIED: Added validation in _render_refinancing_inputs to check
          if refi_year > hold_period.
"""

import streamlit as st
import numpy as np
import pandas as pd # For SOFR curve display
import logging
from typing import Dict, Any, Optional
from propsaber.core.utils import simulation_error_handler

# Use relative imports for components within the propsaber package
from ..core.inputs import SimulationInputs
from ..core.constants import (
    FMT_CURRENCY_ZERO_DP, FMT_INTEGER, FMT_PERCENT_ONE_DP, FMT_PERCENT_TWO_DP,
    FMT_DECIMAL_TWO_DP, LOAN_TYPES, LOAN_TYPE_IO, LOAN_TYPE_AMORT, FLOAT_ATOL
)
# Import the utility function for conversion (if needed, though aggregation happens later now)
# from ..core.utils import convert_to_internal

logger = logging.getLogger(__name__)

# --- Helper to Get Input Value from Session State ---
def get_input_value(key: str, default_value_from_class: Any) -> Any:
    """
    Safely retrieves the current value for an input key from st.session_state['inputs'],
    falling back to the default value from the SimulationInputs class if not found.
    Initializes st.session_state['inputs'] if it doesn't exist.

    Args:
        key: The key corresponding to the SimulationInputs attribute name.
        default_value_from_class: The default value defined in the SimulationInputs dataclass.

    Returns:
        The value from session state or the default value.
    """
    if "inputs" not in st.session_state:
        # This should ideally be initialized once in app.py, but added here as safety
        logger.warning("Initializing st.session_state['inputs'] within get_input_value. Should be done in app.py.")
        st.session_state["inputs"] = {}
    # Return the value from the state dictionary, or the class default if the key is missing
    return st.session_state["inputs"].get(key, default_value_from_class)

# --- Individual Input Rendering Functions ---
@simulation_error_handler
def _render_property_inputs(default_inputs: SimulationInputs):
    """Renders Property Purchase Price and Unit Count inputs."""
    with st.expander("🏠 Property Info", expanded=False):
        st.number_input(
            "Purchase Price ($)", min_value=1.0,
            # Use get_input_value to fetch current state or default
            value=float(get_input_value("purchase_price", default_inputs.purchase_price)),
            step=100000.0, format=FMT_CURRENCY_ZERO_DP, key="input_purchase_price", # Use 'input_' prefix for widget keys
            help="Total acquisition price of the property, inclusive of closing costs."
        )
        st.number_input(
            "Number of Units", min_value=1,
            value=int(get_input_value("num_units", default_inputs.num_units)),
            step=1, format=FMT_INTEGER, key="input_num_units",
            help="Total number of residential units in the property."
        )

@simulation_error_handler
def _render_simulation_setup_inputs(default_inputs: SimulationInputs):
    """Renders input widgets for simulation setup parameters within an expander."""
    with st.expander("⚙️ Simulation Control", expanded=False):
        # Hold Period (Slider, unchanged)
        st.number_input(
            "Hold Period (Years)",
            min_value=1, max_value=30, step=1,
            value=int(get_input_value("hold_period", default_inputs.hold_period)),
            key="input_hold_period",
            help="Investment hold period in years (up to 30 years)."
        )

        # Number of Simulations (Number input, refined step)
        num_sim_help = f"""
        Number of simulation paths to generate. Higher = more accurate but slower.
        Current limit set to {10000:,}.

        Consider 1,000-2,000 for typical analysis, 5,000+ for higher precision.
        """
        st.number_input(
            "Number of Simulations", min_value=10, max_value=10000,
            value=get_input_value("num_simulations", default_inputs.num_simulations),
            step=1000, key="input_num_simulations", format=FMT_INTEGER,
            help=num_sim_help
        )

        # Cyclical Persistence (Number input instead of slider)
        growth_persistence_help = """
        Helps simulate the real-world tendency for market conditions or sentiment to **persist** for a while.
        It controls how much of last year's *unexpected* performance (the random 'shock' affecting Rent, OpEx, etc.) carries over to influence this year's *unexpected* performance.
        * **Low value (0):** Assumes surprises are random flips each year (more jagged path).
        * **High value (near 1):** Assumes surprises tend to continue (smoother, longer cycles).
        * **Default (0.5):** Moderate persistence.
        """
        st.number_input(
            "Cyclical Persistence (Growth)", min_value=0.0, max_value=1.0, step=0.05,
            value=float(get_input_value("growth_persistence_phi", default_inputs.growth_persistence_phi)),
            help=growth_persistence_help,
            key="input_growth_persistence_phi",
            format=FMT_DECIMAL_TWO_DP
        )

@simulation_error_handler
def _render_rent_inputs(default_inputs: SimulationInputs):
    """Renders rent-related inputs using standardized terminology."""
    with st.expander("📈 Rent & Market Regimes", expanded=False):
        st.number_input(
            "Initial Market Rent/Unit ($/Mo)",
            min_value=1.0,
            value=float(get_input_value("base_rent", default_inputs.base_rent)),
            step=1.0, key="input_base_rent", format=FMT_CURRENCY_ZERO_DP,
            help="The actual starting rent per unit per month at Year 0."
        )
        st.markdown("---"); st.subheader("Market Rent vs. Fair Value")

        mrd_help = """
        How far the Initial Market Rent is from the starting Fair Value Rent.
        * **Negative (-5%)**: Initial Rent is 5% *below* Fair Value (potential upside if rents below long-term trend).
        * **Positive (+5%)**: Initial Rent is 5% *above* Fair Value (potential downside if rents above long-term trend).
        * **Zero (0%)**: Initial Rent starts exactly at Fair Value.

        The Projected Market Rent used in cash flows will gradually adjust to meet the Fair Value Rent over the normalization period (entered below).
        See Dynamics tab for graphical representation of simulated results.
        """
        st.number_input(
            "Initial Market Rent Premium/Discount to Fair Value (%)",
            min_value=-20.0, max_value=20.0, step=0.01,
            value=float(get_input_value("market_rent_deviation_pct", default_inputs.market_rent_deviation_pct) * 100.0),
            help=mrd_help,
            key="input_market_rent_deviation_pct",
            format=FMT_PERCENT_TWO_DP
        )

        convergence_help = "How many years it takes for the initial premium/discount (entered above) to fully dissipate, bringing the Projected Market Rent in line with the Fair Value Rent. See Dynamics tab for graphical representation of simulated results."
        st.number_input(
            "Years to Normalize Rent",
            min_value=1, max_value=10, step=1,
            value=int(get_input_value("market_convergence_years", default_inputs.market_convergence_years)),
            help=convergence_help,
            key="input_market_convergence_years",
            format=FMT_INTEGER
        )

        st.markdown("---"); st.subheader("Fair Value Rent Growth (Normal Conditions)")
        normal_growth_help = """
        Expected average annual growth for the Fair Value Rent ONLY during 'Normal' market periods.

        **Note:** This should typically be *higher* than the overall long-term historical average for the market,
        because the historical average includes recessionary periods which are modeled separately below.
        If you don't want to model separate recession effects, set Prob. Normal → Recession (below) to 0% and enter the the projected long-term average here.

        See Dynamics tab for graphical representation of simulated results.
        """
        st.number_input(
            "Avg Fair Value Rent Growth (Normal, %/Yr)",
            min_value=-5.0, max_value=10.0, step=0.05,
            value=float(get_input_value("normal_growth_mean", default_inputs.normal_growth_mean)),
            help=normal_growth_help,
            key="input_normal_growth_mean",
            format=FMT_PERCENT_TWO_DP
        )
        st.number_input(
            "Fair Value Rent Growth Volatility (Normal, % pts)",
            min_value=0.0, max_value=10.0, step=0.01,
            value=float(get_input_value("normal_growth_vol", default_inputs.normal_growth_vol)),
            help="Random fluctuation (standard deviation) around the average Fair Value Rent growth during Normal Conditions. See Dynamics tab for graphical representation of simulated results.",
            key="input_normal_growth_vol",
            format=FMT_PERCENT_TWO_DP
        )

        st.markdown("---"); st.subheader("Fair Value Rent Growth (Recession Conditions)")
        trans_n_r_help = """
        Estimated chance per year of the market entering Recession Conditions from Normal Conditions.
        Set to 0% to effectively disable recession modeling.
        """
        st.number_input(
            "Prob. Normal → Recession (%/Yr)", min_value=0.0, max_value=100.0, step=0.1,
            value=float(get_input_value("transition_normal_to_recession", default_inputs.transition_normal_to_recession) * 100.0),
            help=trans_n_r_help,
            key="input_transition_normal_to_recession",
            format=FMT_PERCENT_ONE_DP
        )
        st.number_input(
            "Avg Fair Value Rent Growth (Recession, %/Yr)",
            min_value=-10.0, max_value=5.0, step=0.01,
            value=float(get_input_value("recession_growth_mean", default_inputs.recession_growth_mean)),
            help="Expected average annual growth for the Fair Value Rent ONLY during Recession Conditions. See Dynamics tab for graphical representation of simulated results.",
            key="input_recession_growth_mean",
            format=FMT_PERCENT_TWO_DP
        )
        st.number_input(
            "Fair Value Rent Growth Volatility (Recession, % pts)",
            min_value=0.0, max_value=15.0, step=0.01,
            value=float(get_input_value("recession_growth_vol", default_inputs.recession_growth_vol)),
            help="Random fluctuation (standard deviation) around the average Fair Value Rent growth during Recession Conditions. See Dynamics tab for graphical representation of simulated results.",
            key="input_recession_growth_vol",
            format=FMT_PERCENT_TWO_DP
        )

        trans_r_n_help = """
        Estimated chance per year of the market recovering to Normal Conditions, *only applicable if currently in Recession Conditions*.
        """
        st.number_input(
            "Prob. Recession → Normal (%/Yr)", min_value=0.0, max_value=100.0, step=0.1,
            value=float(get_input_value("transition_recession_to_normal", default_inputs.transition_recession_to_normal) * 100.0),
            help=trans_r_n_help,
            key="input_transition_recession_to_normal",
            format=FMT_PERCENT_ONE_DP
        )
        # Keep derived calculation unchanged

        # --- Input Summary Calculation (Logic unchanged, text updated) ---
        st.markdown("---")
        st.markdown("**Derived Long-Run Expectation:**")
        try:
            # Read directly from widget state keys
            p_n_to_r_ui = st.session_state.get("input_transition_normal_to_recession", default_inputs.transition_normal_to_recession * 100.0)
            p_r_to_n_ui = st.session_state.get("input_transition_recession_to_normal", default_inputs.transition_recession_to_normal * 100.0)
            g_norm_ui = st.session_state.get("input_normal_growth_mean", default_inputs.normal_growth_mean)
            g_rec_ui = st.session_state.get("input_recession_growth_mean", default_inputs.recession_growth_mean)

            # Use a local conversion helper or import from utils if needed
            def _local_convert(val, is_perc):
                try: return float(val) / 100.0 if is_perc else float(val)
                except: return np.nan

            p_n_to_r = _local_convert(p_n_to_r_ui, True)
            p_r_to_n = _local_convert(p_r_to_n_ui, True)
            g_norm = _local_convert(g_norm_ui, False) / 100.0 # Growth stored as %, convert to decimal
            g_rec = _local_convert(g_rec_ui, False) / 100.0 # Growth stored as %, convert to decimal

            if not all(np.isfinite([p_n_to_r, p_r_to_n, g_norm, g_rec])):
                 raise ValueError("One or more input values for implied growth calculation are invalid.")

            denom = p_n_to_r + p_r_to_n
            if denom > FLOAT_ATOL:
                pi_normal = p_r_to_n / denom
                pi_recession = p_n_to_r / denom
                implied_avg_growth = (pi_normal * g_norm + pi_recession * g_rec) * 100.0
                st.markdown(f"Implied Long-Run Avg Fair Value Rent Growth: `{implied_avg_growth:.1f}%`")
            elif p_n_to_r <= FLOAT_ATOL and p_r_to_n > FLOAT_ATOL:
                 st.markdown(f"Implied Long-Run Avg Fair Value Rent Growth: `{g_norm*100:.1f}%` (Stays in Normal State)")
            elif p_r_to_n <= FLOAT_ATOL and p_n_to_r > FLOAT_ATOL:
                 st.markdown(f"Implied Long-Run Avg Fair Value Rent Growth: `{g_rec*100:.1f}%` (Stays in Recession State)")
            else: # Both transition probs are zero
                 st.markdown(f"Implied Long-Run Avg Fair Value Rent Growth: `{g_norm*100:.1f}%` (No transitions defined)")
        except Exception as e:
            logger.warning(f"Could not calculate implied rent growth summary: {e}")
            st.markdown("*Could not calculate implied long-run average fair value rent growth.*")

@simulation_error_handler
def _render_vacancy_inputs(default_inputs: SimulationInputs):
    """Renders input widgets for vacancy parameters."""
    with st.expander("📉 Vacancy", expanded=False):
        st.number_input(
            "Starting Vacancy Rate (%)", min_value=0.0, max_value=50.0, step=0.01,
            value=float(get_input_value("current_vacancy", default_inputs.current_vacancy) * 100.0),
            help="Initial vacancy rate at the start of the holding period.",
            key="input_current_vacancy",
            format=FMT_PERCENT_TWO_DP
        )
        st.number_input(
            "Target Long-Term Vacancy Rate (%)", min_value=0.0, max_value=30.0, step=0.01,
            value=float(get_input_value("stabilized_vacancy", default_inputs.stabilized_vacancy) * 100.0),
            help="The long-run average vacancy rate the simulation will naturally gravitate towards over time. See Dynamics tab for graphical representation of simulated results.",
            key="input_stabilized_vacancy",
            format=FMT_PERCENT_TWO_DP
        )

        vacancy_reversion_help = """
        Controls how quickly the simulated Vacancy Rate tries to return to the Target Long-Term Vacancy after random fluctuations.
        A higher value means vacancy stabilizes faster (e.g., 1.0 = very fast); a lower value means vacancy drifts more slowly back to the target (e.g., 0.1 = very slow).
        Think of it like the strength of the anchor pulling vacancy back to its average. See Dynamics tab for graphical representation of simulated results.
        """
        st.number_input(
            "Vacancy Reversion Speed", min_value=0.01, max_value=1.0, step=0.01,
            value=float(get_input_value("vacancy_reversion_speed", default_inputs.vacancy_reversion_speed)),
            help=vacancy_reversion_help,
            key="input_vacancy_reversion_speed",
            format=FMT_DECIMAL_TWO_DP
        )

        vacancy_volatility_help = """
        Controls the typical size of random, unpredictable year-to-year jumps or dips in the vacancy rate.
        A higher value represents more market noise or unpredictable swings affecting vacancy. Input as absolute percentage points (e.g., 1.5 means a typical random shock might be around +/- 1.5% vacancy). See Dynamics tab for graphical representation of simulated results.
        """
        st.number_input(
            "Vacancy Rate Volatility (% pts/Yr)", min_value=0.0, max_value=5.0, step=0.01,
            value=float(get_input_value("vacancy_volatility", default_inputs.vacancy_volatility) * 100.0),
            help=vacancy_volatility_help,
            key="input_vacancy_volatility",
            format=FMT_PERCENT_TWO_DP
        )

        st.markdown("---")
        st.markdown("**Derived Vacancy Dynamics:**")
        try:
            reversion_speed_ui = st.session_state.get("input_vacancy_reversion_speed", default_inputs.vacancy_reversion_speed)
            def _local_convert(val, is_perc):
                 try: return float(val) / 100.0 if is_perc else float(val)
                 except: return np.nan
            reversion_speed = _local_convert(reversion_speed_ui, False)

            if np.isfinite(reversion_speed) and reversion_speed > FLOAT_ATOL:
                half_life = np.log(2) / reversion_speed
                st.markdown(f"Approx. Time to Halve Vacancy Gap: `{half_life:.1f} years`")
            elif np.isfinite(reversion_speed):
                 st.markdown("Implied Reversion: `Very slow / No reversion`")
            else:
                 raise ValueError("Invalid reversion speed value.")
        except Exception as e:
            logger.warning(f"Could not calculate vacancy reversion summary: {e}")
            st.markdown("*Could not calculate approx. time to halve vacancy gap.*")

@simulation_error_handler
def _render_other_income_inputs(default_inputs: SimulationInputs):
    """Renders Other Income inputs."""
    with st.expander("📊 Other Income", expanded=False):
        st.number_input(
            "Initial Annual Other Income ($)", min_value=0.0,
            value=float(get_input_value("mean_other_income", default_inputs.mean_other_income)),
            step=100.0, format=FMT_CURRENCY_ZERO_DP, key="input_mean_other_income",
            help="Starting annual income from sources other than rent (e.g., parking, fees, laundry)."
        )
        st.number_input(
            "Avg Other Income Growth (%/Yr)", min_value=-5.0, max_value=10.0, step=0.01,
            value=float(get_input_value("mean_other_income_growth", default_inputs.mean_other_income_growth)),
            help="Expected average annual growth rate for other income sources.",
            key="input_mean_other_income_growth",
            format=FMT_PERCENT_TWO_DP
        )
        st.number_input(
            "Other Income Growth Volatility (% pts)", min_value=0.0, max_value=10.0, step=0.01,
            value=float(get_input_value("other_income_stddev", default_inputs.other_income_stddev)),
            help="Random volatility (standard deviation) around the average other income growth rate.",
            key="input_other_income_stddev",
            format=FMT_PERCENT_TWO_DP
        )

@simulation_error_handler
def _render_opex_inputs(default_inputs: SimulationInputs):
    """Renders Operating Expense inputs."""
    with st.expander("💸 Operating Expenses", expanded=False):
        st.number_input(
            "Initial Annual Operating Expenses ($)", min_value=0.0,
            value=float(get_input_value("mean_expense", default_inputs.mean_expense)),
            step=1000.0, format=FMT_CURRENCY_ZERO_DP, key="input_mean_expense",
            help="Starting total annual operating expenses (excluding CapEx and debt service)."
        )
        st.number_input(
            "Average OpEx Growth (%/Yr)", min_value=0.0, max_value=10.0, step=0.01,
            value=float(get_input_value("mean_expense_growth", default_inputs.mean_expense_growth)),
            help="Expected average annual growth rate for operating expenses.",
            key="input_mean_expense_growth",
            format=FMT_PERCENT_TWO_DP
        )
        st.number_input(
            "OpEx Growth Volatility (% pts)", min_value=0.0, max_value=10.0, step=0.01,
            value=float(get_input_value("expense_stddev", default_inputs.expense_stddev)),
            help="Random volatility (standard deviation) around the average OpEx growth rate.",
            key="input_expense_stddev",
            format=FMT_PERCENT_TWO_DP
        )

@simulation_error_handler
def _render_capex_inputs(default_inputs: SimulationInputs):
    """Renders Capital Expenditures inputs."""
    with st.expander("🛠️ Capital Expenditures (CapEx)", expanded=False):
        st.number_input(
            "Initial Annual CapEx Reserve ($/Unit/Yr)", min_value=0.0,
            value=float(get_input_value("capex_per_unit_yr", default_inputs.capex_per_unit_yr)),
            step=10.0, key="input_capex_per_unit_yr", format=FMT_CURRENCY_ZERO_DP,
            help="Starting annual capital reserve budget per unit."
        )
        st.number_input(
            "Average CapEx Growth (%/Yr)", min_value=0.0, max_value=10.0, step=0.01,
            value=float(get_input_value("mean_capex_growth", default_inputs.mean_capex_growth)),
            help="Expected average annual growth rate for the CapEx budget (e.g., due to inflation).",
            key="input_mean_capex_growth",
            format=FMT_PERCENT_TWO_DP
        )
        st.number_input(
            "CapEx Growth Volatility (% pts)", min_value=0.0, max_value=10.0, step=0.01,
            value=float(get_input_value("capex_stddev", default_inputs.capex_stddev)),
            help="Random volatility (standard deviation) around the average CapEx growth rate.",
            key="input_capex_stddev",
            format=FMT_PERCENT_TWO_DP
        )

@simulation_error_handler
def _render_exit_inputs(default_inputs: SimulationInputs):
    """Renders input widgets for exit assumptions."""
    with st.expander("🚪 Exit Assumptions", expanded=False):
        st.number_input(
            "Average Exit Cap Rate (%)", min_value=1.0, max_value=10.0, step=0.01,
            value=float(get_input_value("mean_exit_cap_rate", default_inputs.mean_exit_cap_rate)),
            help="The expected long-term capitalization rate upon sale, before volatility or adjustments. See Exit tab for graphical representation of simulated results.",
            key="input_mean_exit_cap_rate",
            format=FMT_PERCENT_TWO_DP
        )
        st.number_input(
            "Exit Cap Rate Volatility (% pts)", min_value=0.0, max_value=3.0, step=0.01,
            value=float(get_input_value("exit_cap_rate_stddev", default_inputs.exit_cap_rate_stddev)),
            help="Volatility (standard deviation) around the Average Exit Cap Rate, representing market uncertainty at the time of sale. See Exit tab for graphical representation of simulated results.",
            key="input_exit_cap_rate_stddev",
            format=FMT_PERCENT_TWO_DP
        )

        exit_cap_sens_help = """
        Adjusts the simulated Exit Cap Rate based on the final year's simulated rent growth relative to the long-term Average Fair Value Rent Growth assumption. Use this to model how the market might perceive higher/lower future growth potential at sale.
        * **Negative value (e.g., -0.1):** Higher-than-expected growth *lowers* the cap rate (increases value). E.g., -0.1 means +1% growth surprise lowers cap rate by 0.1% points.
        * **Positive value:** Higher-than-expected growth *increases* the cap rate (decreases value).
        * **Zero:** No adjustment based on terminal rent growth.

        See Dynamics tab for graphical representation of simulated results.
        """
        st.number_input(
            "Exit Cap Adjustment Factor", min_value=-0.5, max_value=0.5, step=0.01,
            value=float(get_input_value("exit_cap_rent_growth_sensitivity", default_inputs.exit_cap_rent_growth_sensitivity)),
            help=exit_cap_sens_help,
            key="input_exit_cap_rent_growth_sensitivity",
            format=FMT_DECIMAL_TWO_DP
        )

        st.number_input(
            "Transaction Cost on Sale (%)", min_value=0.0, max_value=5.0, step=0.01,
            value=float(get_input_value("transaction_cost_pct", default_inputs.transaction_cost_pct) * 100.0),
            help="Costs associated with selling the property (brokerage, legal, etc.) as a % of gross sale price.",
            key="input_transaction_cost_pct",
            format=FMT_PERCENT_TWO_DP
        )
        st.number_input(
            "Gross Exit Value Floor ($)", min_value=0.0,
            value=float(get_input_value("exit_floor_value", default_inputs.exit_floor_value)),
            step=10000.0, format=FMT_CURRENCY_ZERO_DP, key="input_exit_floor_value",
            help="Minimum gross sale price allowed in the simulation, regardless of calculated value based on NOI and cap rate."
        )

@simulation_error_handler
def _render_financing_inputs(
    default_inputs: SimulationInputs,
    forward_curve: Optional[Dict[int, float]],
    std_dev_curve: Optional[Dict[int, float]]
):
    """Renders input widgets for financing parameters."""
    with st.expander("🏦 Financing", expanded=False):
        st.number_input(
            "Loan-to-Cost Ratio (%)", min_value=0.0, max_value=90.0, step=0.01,
            value=float(get_input_value("loan_to_cost", default_inputs.loan_to_cost) * 100.0),
            help="Initial loan amount as a percentage of the purchase price.",
            key="input_loan_to_cost",
            format=FMT_PERCENT_TWO_DP
        )
        st.number_input(
            "Initial Loan Costs (% of Loan)", min_value=0.0, max_value=5.0, step=0.01,
            value=float(get_input_value("initial_loan_costs_pct", default_inputs.initial_loan_costs_pct) * 100.0),
            help="Closing costs for the initial loan (e.g., origination, legal) as a % of the initial loan amount. Reduces Day 0 cash flow.",
            key="input_initial_loan_costs_pct",
            format=FMT_PERCENT_TWO_DP
        )
        st.markdown("---")

        rate_type_options = ["Fixed", "Floating"]
        current_is_variable = get_input_value("is_variable_rate", default_inputs.is_variable_rate)
        default_rate_type_index = 1 if current_is_variable else 0

        rate_type = st.selectbox(
            "Rate Type", options=rate_type_options, index=default_rate_type_index,
            key="input_rate_type", help="Choose between a fixed interest rate or a floating rate based on SOFR forward curve."
        )

        if rate_type == "Fixed":
            st.number_input(
                "Fixed Interest Rate (%/Yr)", min_value=0.0, max_value=12.0, step=0.01,
                value=float(get_input_value("interest_rate", default_inputs.interest_rate) * 100.0),
                help="Annual fixed interest rate for the loan term.",
                key="input_interest_rate",
                format=FMT_PERCENT_TWO_DP
            )
            current_loan_type = get_input_value("loan_type", default_inputs.loan_type)
            if current_loan_type not in LOAN_TYPES: current_loan_type = LOAN_TYPE_IO
            try: loan_type_index = LOAN_TYPES.index(current_loan_type)
            except ValueError: loan_type_index = 0

            st.selectbox(
                "Loan Type", LOAN_TYPES, index=loan_type_index, key="input_loan_type",
                help="Choose between Interest Only or Amortizing loan (Fixed Rate Only)."
            )

            loan_type_state = st.session_state.get("input_loan_type", default_inputs.loan_type)
            if loan_type_state == LOAN_TYPE_AMORT:
                st.number_input(
                    "Amortization Period (Years)", min_value=5, max_value=40, step=1,
                    value=int(get_input_value("loan_term_yrs", default_inputs.loan_term_yrs)),
                    help="Number of years over which the loan amortizes. Loan is held for the hold period, with remaining balance paid at sale.",
                    key="input_loan_term_yrs",
                    format=FMT_INTEGER
                )

        elif rate_type == "Floating":
            if not forward_curve or not std_dev_curve:
                st.warning("SOFR forward curve/std dev data not loaded correctly. Floating rate option unavailable.")
            else:
                st.number_input(
                    "Spread Over SOFR (%/Yr)", min_value=0.0, max_value=5.0, step=0.01,
                    value=float(get_input_value("sofr_spread", default_inputs.sofr_spread) * 100.0),
                    help="Fixed spread added *after* simulating the underlying SOFR component (including floor). See Dynamics tab for graphical representation of simulated results.",
                    key="input_sofr_spread",
                    format=FMT_PERCENT_TWO_DP
                )
                st.number_input(
                    "SOFR Floor (%/Yr)", min_value=0.0, max_value=5.0, step=0.01,
                    value=float(get_input_value("sofr_floor", default_inputs.sofr_floor) * 100.0),
                    help="Minimum level for the simulated SOFR component *before* adding the spread. See Dynamics tab for graphical representation of simulated results.",
                    key="input_sofr_floor",
                    format=FMT_PERCENT_TWO_DP
                )
                volatility_scalar_help = """
                Adjusts the intensity of the random annual volatility applied to the floating rate simulation, using market-implied standard deviations as a base.
                * **1.0 = Use derived volatility directly.**
                * **>1.0 = Amplify volatility (more random rate swings).**
                * **<1.0 = Dampen volatility (less random rate swings).**
                * **0.0 = No random rate volatility (rate follows forward curve + persistence only).**

                See Dynamics tab for graphical representation of simulated results.
                """
                st.number_input(
                    "Floating Rate Volatility Factor",
                    min_value=0.0, max_value=3.0, step=0.01,
                    value=float(get_input_value("volatility_scalar", default_inputs.volatility_scalar)),
                    help=volatility_scalar_help, key="input_volatility_scalar",
                    format=FMT_DECIMAL_TWO_DP
                )
                rate_persistence_help = """
                Helps simulate the tendency for interest rate environments to persist. Controls how much of last year's *unexpected* movement
                in the simulated SOFR component carries over to influence this year's movement.
                * **0 = No persistence:** Rate shocks are independent each year.
                * **Near 1 = High persistence:** Rate shocks create smoother, longer-lasting trends.
                * **0.5 = Moderate persistence:** (Default).

                See Dynamics tab for graphical representation of simulated results.
                """
                st.number_input(
                    "Cyclical Persistence (Rates)",
                    min_value=0.0, max_value=0.95, step=0.01,
                    value=float(get_input_value("rate_persistence_phi", default_inputs.rate_persistence_phi)),
                    help=rate_persistence_help,
                    key="input_rate_persistence_phi",
                    format=FMT_DECIMAL_TWO_DP
                )
                st.caption("Loaded SOFR Forward Curve Preview (First 10 Years):")
                try:
                    if forward_curve:
                        curve_items = sorted(forward_curve.items())
                        curve_df = pd.DataFrame(curve_items, columns=['Year', 'Rate']).head(10)
                        curve_df['Rate'] = curve_df['Rate'].apply(lambda x: f"{x:.2%}" if pd.notna(x) else "N/A")
                        st.dataframe(curve_df, use_container_width=True, height=150)
                    else:
                        st.caption("No forward curve data available for preview.")
                except Exception as e:
                    st.warning(f"Could not display SOFR curve preview: {e}")
                    logger.error(f"SOFR Preview Error: {e}", exc_info=True)
                st.caption("Floating rate loans are modeled as Interest Only.")

@simulation_error_handler
def _render_refinancing_inputs(default_inputs: SimulationInputs):
    """Renders input widgets for refinancing parameters."""
    with st.expander("🔄 Refinancing", expanded=False):
        st.checkbox(
            "Enable Refinancing?",
            value=get_input_value("enable_refinancing", default_inputs.enable_refinancing),
            key="input_enable_refinancing",
            help="If checked, the simulation will attempt a refinance based on the settings below."
        )

        enable_refi_state = st.session_state.get("input_enable_refinancing", default_inputs.enable_refinancing)
        if enable_refi_state:
            st.number_input(
                "Refinance Year",
                min_value=1, max_value=29, step=1,
                value=int(get_input_value("refi_year", default_inputs.refi_year)),
                key="input_refi_year",
                format=FMT_INTEGER,
                help="The specific year during the hold period when the refinance occurs."
            )

            refi_year_value = st.session_state.get("input_refi_year", default_inputs.refi_year)
            hold_period_value = st.session_state.get("input_hold_period", default_inputs.hold_period)
            if int(refi_year_value) > int(hold_period_value):
                st.warning(f"⚠️ Refinance Year ({refi_year_value}) cannot be greater than Hold Period ({hold_period_value}).")

            refi_rate_help = """
            Determines the new fixed interest rate upon refinance.
            Calculated as: Simulated SOFR Rate (in Refi Year) + This Spread.
            This links the refi rate to simulated market conditions based on the current SOFR forward curve.
            """
            st.number_input(
                "New Loan: Fixed Rate Spread to SOFR (%)",
                min_value=1.0, max_value=5.0, step=0.01,
                value=float(get_input_value("refi_fixed_rate_spread_to_sofr", default_inputs.refi_fixed_rate_spread_to_sofr)),
                help=refi_rate_help, key="input_refi_fixed_rate_spread_to_sofr",
                format=FMT_PERCENT_TWO_DP
            )
            refi_ltv_help = """
            Target Loan-to-Value for the new loan, based on the property's simulated value in the Refinance Year.
            Determines cash-out (if New LTV * Value > Old Balance) or cash-in.
            """
            # Sanitize refi_new_ltv value
            raw_ltv_value = get_input_value("refi_new_ltv", default_inputs.refi_new_ltv)
            try:
                ltv_value = float(raw_ltv_value) * 100.0
                if not (50.0 <= ltv_value <= 85.0):
                    ltv_value = float(default_inputs.refi_new_ltv) * 100.0
                    logger.warning(f"Invalid refi_new_ltv value {raw_ltv_value}, resetting to default {default_inputs.refi_new_ltv}")
            except (ValueError, TypeError):
                ltv_value = float(default_inputs.refi_new_ltv) * 100.0
                logger.warning(f"Failed to convert refi_new_ltv value {raw_ltv_value}, using default {default_inputs.refi_new_ltv}")
            st.number_input(
                "New Loan: Target LTV (%)",
                min_value=50.0, max_value=85.0, step=0.01,
                value=ltv_value,
                help=refi_ltv_help, key="input_refi_new_ltv",
                format=FMT_PERCENT_TWO_DP
            )
            st.number_input(
                "New Loan: Amortization Period (Years)",
                min_value=5, max_value=30, step=1,
                value=int(get_input_value("refi_new_amort_period", default_inputs.refi_new_amort_period)),
                help="Amortization term for the new fixed-rate loan taken out at refinance.",
                key="input_refi_new_amort_period", format=FMT_INTEGER
            )
            st.number_input(
                "Refinancing Costs (% of New Loan)",
                min_value=0.0, max_value=5.0, step=0.01,
                value=float(get_input_value("refi_costs_pct_loan", default_inputs.refi_costs_pct_loan) * 100.0),
                help="Costs associated with the refinance (origination, appraisal, etc.) as % of the new loan amount. Reduces cash flow in the refi year.",
                key="input_refi_costs_pct_loan",
                format=FMT_PERCENT_TWO_DP
            )
        else:
            st.caption("Refinancing disabled.")

@simulation_error_handler
def _render_correlation_inputs(default_inputs: SimulationInputs):
    """Renders input widgets for correlation assumptions."""
    with st.expander("🔗 Correlation", expanded=False):
        st.checkbox(
            "Use Correlations Between Shocks",
            value=get_input_value("use_correlations", default_inputs.use_correlations),
            key="input_use_correlations",
            help="If checked, random shocks to Rent, Expenses, Other Income, and Vacancy will be correlated. If unchecked, shocks are independent."
        )

        use_corr_state = st.session_state.get("input_use_correlations", default_inputs.use_correlations)
        if use_corr_state:
            st.caption("Define correlations between annual random shocks:")
            corr_re_help = """
            Correlation between the random shocks affecting Rent growth and Expense growth.
            Positive value models the tendency for periods of surprisingly high Rent growth to also experience surprisingly high Expense growth (e.g., during high inflation). Use 0 if shocks should be independent.
            """
            st.number_input(
                "Correlation: Rent & Expense Shocks", min_value=-1.0, max_value=1.0, step=0.01,
                value=float(get_input_value("corr_rent_expense", default_inputs.corr_rent_expense)),
                help=corr_re_help, key="input_corr_rent_expense", format=FMT_DECIMAL_TWO_DP
            )
            corr_ro_help = """
            Correlation between the random shocks affecting Rent growth and Other Income growth.
            Positive value models tendency for Other Income (parking, fees) to rise when Rents rise unexpectedly. Use 0 if shocks should be independent.
            """
            st.number_input(
                "Correlation: Rent & Other Income Shocks", min_value=-1.0, max_value=1.0, step=0.01,
                value=float(get_input_value("corr_rent_other_income", default_inputs.corr_rent_other_income)),
                help=corr_ro_help, key="input_corr_rent_other_income", format=FMT_DECIMAL_TWO_DP
            )
            corr_rv_help = """
            Correlation between the random shocks affecting Rent growth and Vacancy Rate changes.
            Typically negative, modeling the tendency for Vacancy to decrease when Rents grow surprisingly fast (and vice-versa). Use 0 if shocks should be independent.
            """
            st.number_input(
                "Correlation: Rent & Vacancy Shocks", min_value=-1.0, max_value=1.0, step=0.01,
                value=float(get_input_value("corr_rent_vacancy", default_inputs.corr_rent_vacancy)),
                help=corr_rv_help, key="input_corr_rent_vacancy", format=FMT_DECIMAL_TWO_DP
            )

@simulation_error_handler
def _render_risk_metric_inputs(default_inputs: SimulationInputs):
    """Renders inputs related to risk metric calculations."""
    with st.expander("🛡️ Risk Metric Settings", expanded=False):
        st.number_input(
            "Risk-Free Rate (%/Yr)", min_value=0.0, max_value=10.0, step=0.01,
            value=float(get_input_value("risk_free_rate", default_inputs.risk_free_rate) * 100.0),
            help="Annual risk-free rate used for calculating the Sharpe Ratio.",
            key="input_risk_free_rate",
            format=FMT_PERCENT_TWO_DP
        )
        st.number_input(
            "Hurdle Rate (% IRR)", min_value=0.0, max_value=25.0, step=0.01,
            value=float(get_input_value("hurdle_rate", default_inputs.hurdle_rate) * 100.0),
            help="Target Levered IRR threshold used for calculating the 'Probability Below Hurdle' metric.",
            key="input_hurdle_rate",
            format=FMT_PERCENT_TWO_DP
        )

@simulation_error_handler
def render_sidebar_inputs(
    initial_inputs: Optional[SimulationInputs] = None,
    forward_curve: Optional[Dict[int, float]] = None,
    std_dev_curve: Optional[Dict[int, float]] = None
) -> None:
    """Renders all input sections in the Streamlit sidebar."""
    st.header("📊 Simulation Inputs")
    defaults = initial_inputs if initial_inputs is not None else SimulationInputs()
    if "inputs" not in st.session_state or not st.session_state["inputs"]:
        logger.info("Initializing st.session_state['inputs'] for sidebar rendering.")
        st.session_state["inputs"] = {
            k: getattr(defaults, k)
            for k in SimulationInputs.__annotations__
            if not k.startswith('_') and not isinstance(getattr(SimulationInputs, k, None), property)
        }

    # Render sections
    st.subheader("Property & Settings")
    _render_property_inputs(defaults)
    _render_simulation_setup_inputs(defaults)
    st.markdown("---")
    st.subheader("Operating Assumptions")
    _render_rent_inputs(defaults)
    _render_vacancy_inputs(defaults)
    _render_other_income_inputs(defaults)
    _render_opex_inputs(defaults)
    _render_capex_inputs(defaults)
    st.markdown("---")
    st.subheader("Financing & Exit")
    _render_financing_inputs(defaults, forward_curve, std_dev_curve)
    _render_refinancing_inputs(defaults)
    _render_exit_inputs(defaults)
    st.markdown("---")
    st.subheader("Correlation & Risk")
    _render_correlation_inputs(defaults)
    _render_risk_metric_inputs(defaults)
    st.markdown("---")
