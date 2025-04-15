# app.py
"""
Main Streamlit application entry point for PropSaber.
Orchestrates UI setup (sidebar, tabs), input handling,
simulation execution, and visualization display by calling
functions from the propsaber package modules.

MODIFIED: Fixed NameError in Initial State Snapshot by creating
          SimulationInputs object after input processing and using it.
CORRECTED: Removed non-breaking space causing SyntaxError near main().
CORRECTED: Used 'inputs_used_for_run' in snapshot display after results exist.
MODIFIED: Removed Initial State Snapshot display when no results exist yet.
CORRECTED: Fixed UnboundLocalError in scenario comparison histogram plot.
MODIFIED: Changed scenario comparison plot to be stacked vertically with shared x-axis.
CORRECTED: Fixed TypeError in snapshot calculate_debt_service call signature.
CORRECTED: Input processing loop made more robust to prevent potential double conversion.
"""

import streamlit as st
import os
import time
import numpy as np
import numpy_financial as npf
import pandas as pd
import logging
from pathlib import Path
from datetime import datetime
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from typing import List, Dict, Any, Optional

# --- Basic Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Project Module Imports ---
try:
    from propsaber.core.inputs import SimulationInputs
    from propsaber.core.constants import (
        FLOAT_ATOL, FLOAT_RTOL, DEFAULT_NUM_SIMULATIONS, DEFAULT_HOLD_PERIOD,
        FORWARD_CURVE_PATH, LOAN_TYPE_IO, LOAN_TYPE_AMORT, MONTHS_PER_YEAR
    )
    from propsaber.core.simulation import run_monte_carlo
    from propsaber.core.debt import calculate_debt_service
    from propsaber.core.utils import convert_to_internal, get_valid_paths
    from propsaber.ui.inputs import render_sidebar_inputs
    from propsaber.ui.visualizations import (
        plot_irr_distribution, plot_rent_vs_normal, plot_vacancy_distribution,
        plot_terminal_growth_vs_exit_cap, plot_simulated_sofr_distribution,
        plot_loan_balance_distribution
    )
    from propsaber.data.forward_curve import load_forward_curve_and_std_dev
    from propsaber.scenarios.scenario_manager import (
        save_scenario, load_scenario, list_saved_scenarios, _ensure_scenario_dir_exists
    )
except ImportError as e:
    st.error(f"Failed to import PropSaber modules. Ensure the 'propsaber' package is structured correctly. Error: {e}")
    st.stop()


# --- Page Configuration ---
st.set_page_config(
    page_title="PropSaber Simulation",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Main Application Logic ---
def main():
    st.title("🗡️ PropSaber: Multifamily Simulator")
    st.subheader("Sharper Projections Based on Real-World Uncertainty")

    _ensure_scenario_dir_exists()

    # --- Load Static Data ---
    forward_curve_data, std_dev_curve_data = load_forward_curve_and_std_dev(FORWARD_CURVE_PATH)
    data_loaded_ok = bool(forward_curve_data and std_dev_curve_data)

    # --- Initialize Session State ---
    if "inputs" not in st.session_state:
        logger.info("Initializing st.session_state['inputs'] in app.py")
        default_inputs_instance = SimulationInputs()
        st.session_state["inputs"] = default_inputs_instance.to_dict()
    if "processed_results" not in st.session_state:
        st.session_state["processed_results"] = None
    if "saved_scenarios" not in st.session_state:
        st.session_state["saved_scenarios"] = {}
    if "confirming_delete" not in st.session_state:
         st.session_state.confirming_delete = None


    # --- Sidebar ---
    with st.sidebar:
        run_sim_button = st.button("🚀 Run Simulation", key="run_sim_button", type="primary", use_container_width=True)
        st.markdown("---")
        # This function renders widgets which might update st.session_state immediately via their keys
        render_sidebar_inputs(
            initial_inputs=SimulationInputs(**st.session_state["inputs"]),
            forward_curve=forward_curve_data,
            std_dev_curve=std_dev_curve_data
            )
        # (Scenario Save/Load/Delete logic remains the same)
        st.markdown("---")
        st.subheader("💾 Scenario Files")
        save_name = st.text_input("New Scenario File Name", value="my_scenario", key="save_scenario_name_sidebar", help="Enter name to save CURRENT sidebar inputs to a file.")
        if st.button("Save Inputs to File", key="save_button_sidebar"):
            if save_name: save_scenario(st.session_state["inputs"], filename=f"{save_name}.json") # Save processed state
            else: st.warning("Please enter a file name.")
        st.markdown("---")
        try: scenario_files = list_saved_scenarios()
        except Exception as e: st.error(f"Error reading scenario directory: {e}"); scenario_files = []
        if scenario_files:
            selected_file_load = st.selectbox("Load or Delete Scenario File", options=[""] + scenario_files, index=0, key="scenario_selector_load_sidebar", help="Select a file to load its inputs into the sidebar or delete it.")
            if selected_file_load:
                col_load, col_delete_init = st.columns(2)
                with col_load:
                    if st.button("✅ Load Inputs", key=f"load_button_{selected_file_load}_sidebar"):
                        loaded_inputs_dict = load_scenario(filename=selected_file_load)
                        if loaded_inputs_dict: st.session_state["inputs"] = loaded_inputs_dict; st.session_state["processed_results"] = None; st.success(f"Loaded '{selected_file_load}'."); time.sleep(1.5); st.rerun()
                with col_delete_init:
                    if st.button("🗑️ Delete File", key=f"delete_init_{selected_file_load}_sidebar"): st.session_state.confirming_delete = selected_file_load; st.rerun()
                if st.session_state.get("confirming_delete") == selected_file_load:
                    st.warning(f"Confirm delete: **{selected_file_load}**?"); col_confirm, col_cancel = st.columns(2)
                    with col_confirm:
                        if st.button("✔️ Yes, Delete", key=f"delete_confirm_{selected_file_load}_sidebar"):
                            try: from propsaber.core.constants import SCENARIO_DIR; (SCENARIO_DIR / selected_file_load).unlink(); st.success(f"Deleted: {selected_file_load}"); st.session_state.confirming_delete = None; time.sleep(1); st.rerun()
                            except Exception as e: st.error(f"Error deleting file: {e}"); st.session_state.confirming_delete = None
                    with col_cancel:
                        if st.button("❌ Cancel", key=f"delete_cancel_{selected_file_load}_sidebar"): st.session_state.confirming_delete = None; st.rerun()
        else: st.info("No saved scenario files found.")


    # --- Input Processing & Validation ---
    processed_inputs_dict = {}
    inputs_valid_for_processing = True
    default_inputs_instance = SimulationInputs()
    percentage_decimal_keys = {"market_rent_deviation_pct", "transition_normal_to_recession", "transition_recession_to_normal", "current_vacancy", "stabilized_vacancy", "vacancy_volatility", "loan_to_cost", "interest_rate", "transaction_cost_pct", "risk_free_rate", "hurdle_rate", "sofr_spread", "sofr_floor", "initial_loan_costs_pct", "refi_new_ltv", "refi_costs_pct_loan"}
    percentage_direct_keys = {"normal_growth_mean", "normal_growth_vol", "recession_growth_mean", "recession_growth_vol", "mean_other_income_growth", "other_income_stddev", "mean_expense_growth", "expense_stddev", "mean_capex_growth", "capex_stddev", "mean_exit_cap_rate", "exit_cap_rate_stddev", "refi_fixed_rate_spread_to_sofr"}
    all_input_keys = {k for k in SimulationInputs.__annotations__ if not k.startswith('_') and not isinstance(getattr(SimulationInputs, k, None), property)}

    for key in all_input_keys: # Process inputs from widgets/state
        widget_key = f"input_{key}"; default_value = getattr(default_inputs_instance, key, None); default_type = type(default_value) if default_value is not None else None

        # --- <<< CHANGE START: More robust ui_value retrieval >>> ---
        # Prioritize direct widget state if it exists, otherwise use current processed state (or default if none)
        if widget_key in st.session_state:
            ui_value = st.session_state[widget_key]
            # logger.debug(f"Processing key '{key}': Found value in widget state ('{widget_key}'): {ui_value}")
        else:
            # Fallback to existing processed value or default if widget hasn't rendered yet
            ui_value = st.session_state.get("inputs", {}).get(key, default_value)
            # logger.debug(f"Processing key '{key}': Using value from existing state/default: {ui_value}")
        # --- <<< CHANGE END >>> ---

        # Handle special cases derived from other inputs
        if key == 'is_variable_rate': processed_inputs_dict[key] = (st.session_state.get("input_rate_type", "Fixed") == "Floating"); continue
        if key == 'loan_type':
            # Use already processed 'is_variable_rate' if available in this pass, else check widget state
            is_float = processed_inputs_dict.get('is_variable_rate', (st.session_state.get("input_rate_type", "Fixed") == "Floating"))
            if is_float: processed_inputs_dict[key] = LOAN_TYPE_IO
            else: processed_inputs_dict[key] = st.session_state.get("input_loan_type", default_inputs_instance.loan_type)
            continue
        if ui_value is None: processed_inputs_dict[key] = default_value; continue

        # Convert UI value to internal format
        try:
            if key in percentage_decimal_keys: processed_inputs_dict[key] = convert_to_internal(ui_value, True)
            elif key in percentage_direct_keys: processed_inputs_dict[key] = convert_to_internal(ui_value, False)
            elif default_type == int: processed_inputs_dict[key] = int(round(float(ui_value)))
            elif default_type == float: processed_inputs_dict[key] = float(ui_value)
            elif default_type == bool:
                 if isinstance(ui_value, str): processed_inputs_dict[key] = ui_value.lower() == 'true'
                 else: processed_inputs_dict[key] = bool(ui_value)
            else: processed_inputs_dict[key] = ui_value
        except Exception as e: logger.error(f"Error processing input key '{key}': {e}"); processed_inputs_dict[key] = default_value; inputs_valid_for_processing = False
    processed_inputs_dict["refi_new_ltv"] = st.session_state.get(
        "input_refi_new_ltv",
        default_inputs_instance.refi_new_ltv * 100.0  # fallback to default and convert to percentage first
    ) / 100.0

    processed_inputs_dict["refi_costs_pct_loan"] = st.session_state.get(
        "input_refi_costs_pct_loan",
        default_inputs_instance.refi_costs_pct_loan * 100.0  # fallback to default percentage
    ) / 100.0


    # --- Update session state only if necessary ---
    current_session_inputs_dict = st.session_state.get("inputs", {})
    if not all(np.isclose(processed_inputs_dict.get(k, np.nan), current_session_inputs_dict.get(k, np.nan), rtol=FLOAT_RTOL, atol=FLOAT_ATOL, equal_nan=True) if isinstance(processed_inputs_dict.get(k), float) else processed_inputs_dict.get(k) == current_session_inputs_dict.get(k) for k in all_input_keys):
         logger.info("Processed inputs differ from session state. Updating st.session_state['inputs'].")
         st.session_state["inputs"] = processed_inputs_dict.copy()


    # --- Create SimulationInputs Object ---
    sim_inputs_obj: Optional[SimulationInputs] = None
    inputs_valid_for_snapshot = False # Flag specifically for snapshot validity
    try:
        sim_inputs_obj = SimulationInputs(**st.session_state["inputs"])
        inputs_valid_for_snapshot = True
        logger.info("Successfully created SimulationInputs object from session state.")
    except Exception as e:
        # Display error prominently if object creation fails
        st.error(f"Input Configuration Error: {e}. Cannot proceed. Please check sidebar inputs.")
        logger.error(f"Failed to create SimulationInputs object from session state: {e}", exc_info=True)
        # Keep inputs_valid_for_snapshot as False


    # --- Simulation Execution ---
    processed_results = st.session_state.get("processed_results")
         
    if run_sim_button:
        logger.info("Run Simulation button clicked.")
        st.session_state["processed_results"] = None
        processed_results = None

        inputs_valid_for_sim = True  # Assume valid until checks fail

        # Check initial object validity
        if sim_inputs_obj and inputs_valid_for_snapshot:
            sim_inputs_to_run = sim_inputs_obj
            num_sims_run = sim_inputs_to_run.num_simulations
            logger.info("Using pre-validated SimulationInputs object for simulation run.")
        else:
            st.error("Cannot run simulation: Input configuration errors identified earlier.")
            logger.error("Simulation run aborted due to invalid input object creation.")
            inputs_valid_for_sim = False

        # Ensure data loaded correctly
        if not data_loaded_ok:
            st.error("Cannot run simulation: Forward curve data failed to load.")
            inputs_valid_for_sim = False

        # Begin explicit logical validation checks
        validation_errors = []

        if inputs_valid_for_sim:
            if sim_inputs_to_run.enable_refinancing:
                if sim_inputs_to_run.refi_year > sim_inputs_to_run.hold_period:
                    validation_errors.append(
                        f"Refinance Year ({sim_inputs_to_run.refi_year}) cannot exceed Hold Period ({sim_inputs_to_run.hold_period})."
                    )

        if validation_errors:
            for error_msg in validation_errors:
                st.error(f"Input Error: {error_msg}")
                logger.warning(f"Validation failed: {error_msg}")
            inputs_valid_for_sim = False

        # Execute simulation if inputs are valid
        if inputs_valid_for_sim:
            with st.spinner(f"Running {num_sims_run} simulations..."):
                mc_results = run_monte_carlo(
                    inputs=sim_inputs_to_run,
                    num_simulations=num_sims_run,
                    forward_curve=forward_curve_data,
                    std_dev_curve=std_dev_curve_data
                )

            if mc_results is None or 'error' in mc_results:
                error_message = mc_results.get('error', 'Unknown simulation error.') if mc_results else 'Critical error.'
                st.error(f"Simulation Error: {error_message}")
                logger.error(f"Simulation execution failed: {error_message}")
                st.session_state["processed_results"] = None
            else:
                st.session_state["processed_results"] = mc_results
                logger.info("Simulation finished successfully.")

                # Immediately unpack results into variables
                processed_results = mc_results
                metrics = processed_results.get("metrics", {})
                risk_metrics = processed_results.get("risk_metrics", {})
                plot_data = processed_results.get("plot_data", {})
                avg_cash_flow_data = plot_data.get("avg_cash_flows", {})
                scatter_plot_data = plot_data.get("scatter_plot", {})
                finite_unlevered = processed_results.get("finite_unlevered_irrs", [])
                finite_levered = processed_results.get("finite_levered_irrs", [])
                finite_exit_values = processed_results.get("finite_exit_values", [])
                finite_exit_caps = processed_results.get("finite_exit_caps", [])
                sim_results_completed_audit = processed_results.get("raw_results_for_audit", [])

                # Ensure input objects are created again for display
                try:
                    inputs_used_for_run = SimulationInputs(**st.session_state['inputs'])
                    inputs_valid_for_display = True
                except Exception as e:
                    st.error(f"Error recreating inputs object for display: {e}")
                    inputs_valid_for_display = False
                    inputs_used_for_run = None

                # Define and populate UI tabs
                tab_keys = ["📊 Summary", "📈 IRR", "💰 Pro-Forma", "📉 Dynamics", "🛡️ Risk", "🔍 Audit", "🚪 Exit", "🔎 Sensitivity", "🗂️ Scenarios", "ℹ️ Guide"]
                tabs = st.tabs(tab_keys)

                with tabs[tab_keys.index("📊 Summary")]:
                    st.subheader("Key Performance Indicators (KPIs)")
                    col_kp1, col_kp2, col_kp3 = st.columns(3)
                    mean_l_irr = metrics.get("mean_levered_irr", np.nan)
                    median_l_irr = metrics.get("median_levered_irr", np.nan)
                    p05_l_irr = metrics.get("p05_levered_irr", np.nan)

                    col_kp1.metric("Mean Levered IRR", f"{mean_l_irr:.1%}" if np.isfinite(mean_l_irr) else "N/A")
                    col_kp1.metric("Median Levered IRR", f"{median_l_irr:.1%}" if np.isfinite(median_l_irr) else "N/A")
                    col_kp1.metric("5th Pctl Levered IRR (VaR 95%)", f"{p05_l_irr:.1%}" if np.isfinite(p05_l_irr) else "N/A")

                    mean_exit_val = metrics.get("mean_exit_value", np.nan)
                    mean_exit_cap = metrics.get("mean_exit_cap", np.nan)

                    col_kp2.metric("Mean Net Exit Value", f"${mean_exit_val:,.0f}" if np.isfinite(mean_exit_val) else "N/A")
                    col_kp2.metric("Mean Exit Cap Rate", f"{mean_exit_cap*100:.2f}%" if np.isfinite(mean_exit_cap) else "N/A")

                    prob_loss = risk_metrics.get("Prob. Loss (IRR < 0%)", np.nan)
                    prob_hurdle = risk_metrics.get("Prob. Below Hurdle", np.nan)

                    hurdle_rate_disp = getattr(inputs_used_for_run, 'hurdle_rate', np.nan) if inputs_used_for_run else np.nan
                    hurdle_label = f"Prob < {hurdle_rate_disp:.0%} Hurdle" if np.isfinite(hurdle_rate_disp) else "Prob < Hurdle"

                    col_kp3.metric("Prob. Loss (IRR < 0%)", f"{prob_loss:.1%}" if np.isfinite(prob_loss) else "N/A")
                    col_kp3.metric(hurdle_label, f"{prob_hurdle:.1%}" if np.isfinite(prob_hurdle) else "N/A")
                    sharpe = risk_metrics.get("Sharpe Ratio", np.nan)
                    col_kp3.metric("Sharpe Ratio", f"{sharpe:.2f}" if np.isfinite(sharpe) else "N/A")

            # Display Initial State Snapshot (AFTER results exist)
            st.markdown("---"); st.subheader("Year 0 / Initial State Snapshot (Based on Run Inputs)")
            if inputs_used_for_run and inputs_valid_for_display:
                try:
                    # Calculate initial monthly payment based on inputs used for the run
                    initial_monthly_payment_snapshot = 0.0
                    if not inputs_used_for_run.is_variable_rate and inputs_used_for_run.loan_type == LOAN_TYPE_AMORT:
                         rate_m = inputs_used_for_run.interest_rate / MONTHS_PER_YEAR; periods = inputs_used_for_run.loan_term_yrs * MONTHS_PER_YEAR; loan_amt = inputs_used_for_run.loan_amount
                         if periods > 0 and loan_amt > FLOAT_ATOL:
                              if abs(rate_m) > FLOAT_ATOL:
                                  try: initial_monthly_payment_snapshot = npf.pmt(rate_m, periods, -loan_amt)
                                  except Exception: initial_monthly_payment_snapshot = 0.0
                              else: initial_monthly_payment_snapshot = loan_amt / periods
                         if not np.isfinite(initial_monthly_payment_snapshot): initial_monthly_payment_snapshot = 0.0

                    potential_gross_rent = inputs_used_for_run.num_units * inputs_used_for_run.base_rent * 12
                    vacancy_amount = potential_gross_rent * inputs_used_for_run.current_vacancy; effective_gross_rent = potential_gross_rent - vacancy_amount
                    other_income_val = inputs_used_for_run.mean_other_income; effective_gross_income = effective_gross_rent + other_income_val
                    op_ex = inputs_used_for_run.mean_expense; net_operating_income = inputs_used_for_run.initial_noi
                    initial_cap_rate = net_operating_income / inputs_used_for_run.purchase_price if inputs_used_for_run.purchase_price > FLOAT_ATOL else np.nan
                    capex_yr1 = inputs_used_for_run.initial_capex; unlevered_cf_yr1_approx = net_operating_income - capex_yr1
                    loan_amount_val = inputs_used_for_run.loan_amount

                    # --- <<< CORRECTED CALL >>> ---
                    interest_yr1, principal_yr1, _, effective_rate_yr1, _ = calculate_debt_service(
                        current_loan_type=inputs_used_for_run.loan_type,
                        current_interest_rate=inputs_used_for_run.interest_rate,
                        current_is_variable_rate=inputs_used_for_run.is_variable_rate,
                        current_balance=loan_amount_val,
                        monthly_payment=initial_monthly_payment_snapshot, # Pass calculated payment
                        year=1,
                        # Pass remaining args required by debt.py's function signature
                        sofr_spread=inputs_used_for_run.sofr_spread,
                        forward_curve=forward_curve_data,
                        std_dev_curve=std_dev_curve_data,
                        sofr_floor=inputs_used_for_run.sofr_floor,
                        rate_persistence_phi=inputs_used_for_run.rate_persistence_phi,
                        volatility_scalar=inputs_used_for_run.volatility_scalar,
                        prev_volatile_sofr_comp=None
                        # Removed incorrect/unnecessary args: current_loan_amount, current_loan_term_yrs
                    )
                    # --- <<< END CORRECTED CALL >>> ---

                    if not np.isfinite(interest_yr1): interest_yr1 = 0.0;
                    if not np.isfinite(principal_yr1): principal_yr1 = 0.0
                    total_debt_service_yr1 = interest_yr1 + principal_yr1; levered_cf_yr1_approx = unlevered_cf_yr1_approx - total_debt_service_yr1
                    initial_equity = inputs_used_for_run.initial_equity; levered_cash_yield = levered_cf_yr1_approx / initial_equity if initial_equity > FLOAT_ATOL else np.nan

                    col_is1, col_is2 = st.columns(2)
                    # (Display metrics code remains the same)
                    with col_is1: st.metric("Purchase Price", f"${inputs_used_for_run.purchase_price:,.0f}"); st.metric("Potential Gross Rent (PGR)", f"${potential_gross_rent:,.0f}"); st.metric(f"Less: Vacancy ({inputs_used_for_run.current_vacancy:.1%})", f"(${vacancy_amount:,.0f})"); st.metric("Effective Gross Rent (EGR)", f"${effective_gross_rent:,.0f}"); st.metric("Plus: Other Income", f"${other_income_val:,.0f}"); st.metric("Effective Gross Income (EGI)", f"${effective_gross_income:,.0f}"); st.metric("Less: Operating Expenses", f"(${op_ex:,.0f})"); st.metric("Net Operating Income (NOI)", f"${net_operating_income:,.0f}"); st.metric("Initial Going-In Cap Rate", f"{initial_cap_rate:.2%}" if np.isfinite(initial_cap_rate) else "N/A")
                    with col_is2: st.metric("Less: CapEx (Yr 1 Est.)", f"(${capex_yr1:,.0f})"); st.metric("Unlevered CF (Yr 1 Est.)", f"${unlevered_cf_yr1_approx:,.0f}"); st.markdown("---"); st.metric(f"Loan Amount ({inputs_used_for_run.loan_to_cost:.0%})", f"${loan_amount_val:,.0f}"); st.metric("Initial Equity", f"${initial_equity:,.0f}"); st.metric(f"Interest Rate (Yr 1)", f"{effective_rate_yr1:.2%}" if np.isfinite(effective_rate_yr1) else "N/A"); st.metric("Less: Debt Service (Yr 1 Est.)", f"(${total_debt_service_yr1:,.0f})"); st.metric("Levered CF (Yr 1 Est.)", f"${levered_cf_yr1_approx:,.0f}"); st.metric("Levered Cash Yield (Yr 1 Est.)", f"{levered_cash_yield:.1%}" if np.isfinite(levered_cash_yield) else "N/A")
                except Exception as e:
                    st.error(f"Error calculating Initial State Snapshot: {e}"); logger.error(f"Initial State Calc Error after run: {e}", exc_info=True)
            else:
                 st.warning("Cannot display Initial State Snapshot as input object could not be recreated.")


        # --- Other Tabs (IRR, Pro-Forma, Dynamics, Risk, Audit, Exit, Sensitivity, Scenarios, Guide) ---
        with tabs[tab_keys.index("📈 IRR")]:
             st.subheader("IRR Distribution Analysis")
             mean_unlevered_irr = metrics.get("mean_unlevered_irr", np.nan); median_unlevered_irr = metrics.get("median_unlevered_irr", np.nan); p05_unlevered_irr = metrics.get("p05_unlevered_irr", np.nan); p95_unlevered_irr = metrics.get("p95_unlevered_irr", np.nan); mean_l_irr = metrics.get("mean_levered_irr", np.nan); median_l_irr = metrics.get("median_levered_irr", np.nan); p05_l_irr = metrics.get("p05_levered_irr", np.nan); p95_l_irr = metrics.get("p95_levered_irr", np.nan)
             col1, col2, col3, col4 = st.columns(4); col1.metric("Mean Unlevered IRR", f"{mean_unlevered_irr:.1%}" if np.isfinite(mean_unlevered_irr) else "N/A"); col2.metric("Median Unlevered IRR", f"{median_unlevered_irr:.1%}" if np.isfinite(median_unlevered_irr) else "N/A"); col3.metric("5th Pctl Unlevered IRR", f"{p05_unlevered_irr:.1%}" if np.isfinite(p05_unlevered_irr) else "N/A"); col4.metric("95th Pctl Unlevered IRR", f"{p95_unlevered_irr:.1%}" if np.isfinite(p95_unlevered_irr) else "N/A")
             col5, col6, col7, col8 = st.columns(4); col5.metric("Mean Levered IRR", f"{mean_l_irr:.1%}" if np.isfinite(mean_l_irr) else "N/A"); col6.metric("Median Levered IRR", f"{median_l_irr:.1%}" if np.isfinite(median_l_irr) else "N/A"); col7.metric("5th Pctl Levered IRR", f"{p05_l_irr:.1%}" if np.isfinite(p05_l_irr) else "N/A"); col8.metric("95th Pctl Levered IRR", f"{p95_l_irr:.1%}" if np.isfinite(p95_l_irr) else "N/A")
             st.markdown("---"); st.subheader("IRR Distributions"); unlev_failures = processed_results.get("unlev_forced_failures", 0); unlev_title = f"Distribution of Unlevered IRRs ({len(finite_unlevered)} valid runs)";
             if unlev_failures > 0: unlev_title += f" — {unlev_failures} failures"
             if finite_unlevered: fig_unlevered = plot_irr_distribution(irr_values=finite_unlevered, title=unlev_title, mean_irr=mean_unlevered_irr, median_irr=median_unlevered_irr, p05_irr=p05_unlevered_irr, p95_irr=p95_unlevered_irr, color='blue', percent_format=True); st.plotly_chart(fig_unlevered, use_container_width=True)
             else: st.warning('Unlevered IRR data not available.')
             st.markdown("---"); lev_failures = processed_results.get("lev_forced_failures", 0); lev_title = f"Distribution of Levered IRRs ({len(finite_levered)} valid runs)";
             if lev_failures > 0: lev_title += f" — {lev_failures} failures"
             if finite_levered: # Levered IRR plots...
                 st.markdown("#### Levered IRR Distribution (Full View)"); fig_levered_full = plot_irr_distribution(irr_values=finite_levered, title=lev_title, mean_irr=mean_l_irr, median_irr=median_l_irr, p05_irr=p05_l_irr, p95_irr=p95_l_irr, color='orange', percent_format=True); st.plotly_chart(fig_levered_full, use_container_width=True)
                 st.markdown("---"); st.markdown("#### Zoomed-In View of Levered IRRs"); x_range_zoom = None; zoom_title = "Zoomed View: Levered IRRs (Error Calculating Range)"
                 try: # Zoom logic...
                     if len(finite_levered) >= 2:
                         low_bound = np.percentile(finite_levered, 1)
                         high_bound = np.percentile(finite_levered, 99)
                         zoom_min = max(-0.25, low_bound - 0.05)
                         zoom_max = min(0.75, high_bound + 0.05)
                         if zoom_max > zoom_min + 0.01:
                             x_range_zoom = (zoom_min, zoom_max)
                             zoom_title = f"Zoomed View: Levered IRRs ({x_range_zoom[0]:.0%} to {x_range_zoom[1]:.0%})"
                         else: # Inner else
                             st.caption("Zoomed IRR plot skipped: Range too narrow.")
                     # *** CORRECTED INDENTATION FOR THIS ELSE ***
                     else: # Outer else (for len < 2)
                         st.caption("Zoomed IRR plot skipped: Not enough data.")
                     # *** END OF CORRECTION ***

                     # Plotting logic (still inside try block)
                     if x_range_zoom:
                         fig_levered_zoom = plot_irr_distribution(
                             irr_values=finite_levered, title=zoom_title,
                             mean_irr=mean_l_irr, median_irr=median_l_irr,
                             p05_irr=p05_l_irr, p95_irr=metrics.get('p95_levered_irr', np.nan),
                             color='coral', percent_format=True, x_range=x_range_zoom, bins=30
                         )
                         st.plotly_chart(fig_levered_zoom, use_container_width=True)
                 except Exception as e: # <<< The except block for the try >>>
                     logger.error(f"Error creating zoomed IRR plot: {e}")
                     st.caption(f"Could not generate zoomed IRR plot: {e}")
             else: st.warning('Levered IRR data not available.')


        with tabs[tab_keys.index("💰 Pro-Forma")]:
            st.subheader("Average Annual Pro-Forma Cash Flows")
            hold_period_actual = inputs_used_for_run.hold_period

            if not avg_cash_flow_data or not avg_cash_flow_data.get("noi") or len(avg_cash_flow_data["noi"]) != hold_period_actual or any(pd.isna(x) for x in avg_cash_flow_data["noi"]):
                st.info("Average cash flow data is missing or incomplete for Pro-Forma.")
            else:
                purchase_price = inputs_used_for_run.purchase_price
                loan_to_cost = inputs_used_for_run.loan_to_cost
                initial_loan_proceeds = purchase_price * loan_to_cost
                initial_loan_costs = initial_loan_proceeds * inputs_used_for_run.initial_loan_costs_pct
                net_initial_loan_proceeds = initial_loan_proceeds - initial_loan_costs
                initial_equity = purchase_price - net_initial_loan_proceeds

                refi_year = inputs_used_for_run.refi_year if inputs_used_for_run.enable_refinancing else None
                # Calculate refinancing proceeds: new loan amount - costs (not subtracting payoff)
                refi_proceeds_net = 0.0
                if refi_year:
                    noi_at_refi = avg_cash_flow_data.get("noi", [0.0] * hold_period_actual)[refi_year - 1]
                    refi_cap_rate = inputs_used_for_run.mean_exit_cap_rate / 100.0
                    if noi_at_refi > 0 and refi_cap_rate > FLOAT_ATOL:
                        property_value = noi_at_refi / refi_cap_rate
                        new_loan_amount = property_value * inputs_used_for_run.refi_new_ltv
                        refi_costs = new_loan_amount * inputs_used_for_run.refi_costs_pct_loan
                        refi_proceeds_net = new_loan_amount - refi_costs
                    else:
                        logger.warning(f"Cannot calculate refinancing proceeds: NOI={noi_at_refi}, CapRate={refi_cap_rate}")
                # Loan payoff at refi is the loan balance at the start of the refi year
                refi_loan_payoff = avg_cash_flow_data.get("loan_balance", [0.0] * hold_period_actual)[refi_year - 2] if refi_year and refi_year > 1 else 0.0

                net_sale_proceeds = metrics.get("mean_exit_value", 0.0)
                final_loan_payoff = metrics.get("mean_loan_payoff", avg_cash_flow_data.get("loan_balance", [0.0])[-1] if avg_cash_flow_data.get("loan_balance") else 0.0)

                # Initialize rows for Net Loan Proceeds and Loan Payoff
                net_loan_proceeds_row = [net_initial_loan_proceeds] + [0.0] * hold_period_actual
                loan_payoff_row = [0.0] * (hold_period_actual + 1)

                if refi_year:
                    net_loan_proceeds_row[refi_year] = refi_proceeds_net
                    loan_payoff_row[refi_year] = -refi_loan_payoff

                loan_payoff_row[-1] = -final_loan_payoff  # Final loan payoff at sale

                last_avg_unlev_cf = avg_cash_flow_data.get("unlevered_cf", [0.0])[-1] if avg_cash_flow_data.get("unlevered_cf") else 0.0
                last_avg_lev_cf = avg_cash_flow_data.get("levered_cf", [0.0])[-1] if avg_cash_flow_data.get("levered_cf") else 0.0
                final_avg_unlev_cf = last_avg_unlev_cf + net_sale_proceeds
                final_avg_lev_cf = last_avg_lev_cf + net_sale_proceeds - final_loan_payoff

                total_years = hold_period_actual + 1

                proforma_rows = [
                    ("--- Unlevered Cash Flows ---", [np.nan] * total_years),
                    ("Purchase Price", [-purchase_price] + [np.nan] * hold_period_actual),
                    ("--- Operations ---", [np.nan] * total_years),
                    ("Potential Gross Rent (PGR)", [np.nan] + avg_cash_flow_data.get("potential_rent", [])),
                    ("Less: Vacancy Loss", [np.nan] + [-abs(v) for v in avg_cash_flow_data.get("vacancy_loss", [])]),
                    ("Effective Gross Rent (EGR)", [np.nan] + avg_cash_flow_data.get("egr", [])),
                    ("Plus: Other Income", [np.nan] + avg_cash_flow_data.get("other_income", [])),
                    ("Effective Gross Income (EGI)", [np.nan] + avg_cash_flow_data.get("egi", [])),
                    ("Less: Operating Expenses", [np.nan] + [-abs(e) for e in avg_cash_flow_data.get("expenses", [])]),
                    ("Net Operating Income (NOI)", [np.nan] + avg_cash_flow_data.get("noi", [])),
                    ("Less: Capital Expenditures (CapEx)", [np.nan] + [-abs(c) for c in avg_cash_flow_data.get("capex", [])]),
                    ("Unlevered Cash Flow (Op.)", [np.nan] + avg_cash_flow_data.get("unlevered_cf", [])),
                    ("--- Levered Cash Flows ---", [np.nan] * total_years),
                    ("Net Loan Proceeds", net_loan_proceeds_row),
                    ("Loan Payoff", loan_payoff_row),
                    ("--- Debt Service ---", [np.nan] * total_years),
                    ("Interest Paid", [np.nan] + [-abs(i) for i in avg_cash_flow_data.get("interest", [])]),
                    ("Principal Paid", [np.nan] + [-abs(p) for p in avg_cash_flow_data.get("principal", [])]),
                    ("Levered Cash Flow (Op.)", [np.nan] + avg_cash_flow_data.get("levered_cf", [])),
                    ("--- Sale ---", [np.nan] * total_years),
                    ("Net Sale Proceeds", [np.nan] * hold_period_actual + [net_sale_proceeds]),
                    ("--- Cash Flows for IRR ---", [np.nan] * total_years),
                    ("Unlevered Cash Flow (IRR)", [-purchase_price] + avg_cash_flow_data.get("unlevered_cf", [0.0] * hold_period_actual)[:-1] + [final_avg_unlev_cf]),
                    ("Levered Cash Flow (IRR)", [-initial_equity] + avg_cash_flow_data.get("levered_cf", [0.0] * hold_period_actual)[:-1] + [final_avg_lev_cf]),
                    ("--- Other Info ---", [np.nan] * total_years),
                    ("End of Year Loan Balance", [net_initial_loan_proceeds] + avg_cash_flow_data.get("loan_balance", []))
                ]

                clean_rows = [
                    (label, vals if isinstance(vals, list) and len(vals) == total_years else [np.nan] * total_years)
                    for label, vals in proforma_rows
                ]

                proforma_df = pd.DataFrame.from_dict(
                    {label: vals for label, vals in clean_rows},
                    orient="index",
                    columns=[f"Year {i}" for i in range(total_years)]
                )

                def fmt_proforma(val):
                    if pd.isna(val):
                        return "-"
                    try:
                        num_val = float(val)
                        if abs(num_val) >= 1:
                            return f"$({abs(num_val):,.0f})" if num_val < 0 else f"${num_val:,.0f}"
                        elif abs(num_val) > 1e-3:
                            return f"$({abs(num_val):,.2f})" if num_val < 0 else f"${num_val:,.2f}"
                        else:
                            return "$0"
                    except (TypeError, ValueError):
                        return str(val)

                st.dataframe(
                    proforma_df.style.format(fmt_proforma, na_rep="-")
                    .set_properties(**{"text-align": "right"})
                    .set_table_styles([{"selector": "th", "props": [("text-align", "left")]}]),
                    use_container_width=True
                )

                st.markdown("##### IRR Calculated from Average Cash Flow Stream")
                avg_unlev_irr_display = "N/A"
                avg_lev_irr_display = "N/A"
                try:
                    avg_unlev_stream_vals = proforma_df.loc["Unlevered Cash Flow (IRR)"].astype(float).tolist()
                    avg_lev_stream_vals = proforma_df.loc["Levered Cash Flow (IRR)"].astype(float).tolist()
                    # Debug: Log the streams
                    logger.info(f"Unlevered Cash Flow Stream: {avg_unlev_stream_vals}")
                    logger.info(f"Levered Cash Flow Stream: {avg_lev_stream_vals}")
                    # Check if streams are valid for IRR
                    if len(avg_unlev_stream_vals) >= 2 and all(np.isfinite(avg_unlev_stream_vals)) and any(x < 0 for x in avg_unlev_stream_vals) and any(x > 0 for x in avg_unlev_stream_vals):
                        avg_unlev_irr = npf.irr(avg_unlev_stream_vals)
                        if np.isfinite(avg_unlev_irr):
                            avg_unlev_irr_display = f"{avg_unlev_irr:.1%}"
                        else:
                            logger.warning("Unlevered IRR calculation returned non-finite value.")
                    else:
                        logger.warning("Unlevered cash flow stream invalid for IRR calculation: insufficient length, non-finite values, or no sign changes.")
                    
                    if len(avg_lev_stream_vals) >= 2 and all(np.isfinite(avg_lev_stream_vals)) and any(x < 0 for x in avg_lev_stream_vals) and any(x > 0 for x in avg_lev_stream_vals):
                        avg_lev_irr = npf.irr(avg_lev_stream_vals)
                        if np.isfinite(avg_lev_irr):
                            avg_lev_irr_display = f"{avg_lev_irr:.1%}"
                        else:
                            logger.warning("Levered IRR calculation returned non-finite value.")
                    else:
                        logger.warning("Levered cash flow stream invalid for IRR calculation: insufficient length, non-finite values, or no sign changes.")
                except Exception as e:
                    logger.error(f"Error calculating average IRR: {e}")

                col1, col2 = st.columns(2)
                col1.metric("Unlevered IRR (Avg CF)", avg_unlev_irr_display)
                col2.metric("Levered IRR (Avg CF)", avg_lev_irr_display)

                # Key assumptions table
                st.markdown("---")
                st.subheader("Key Assumptions & Metrics (Average by Year)")
                exit_cap_rate = metrics.get("mean_exit_cap", np.nan)
                assumption_rows = [
                    ("Avg Market Rent ($/Unit/Mo)", avg_cash_flow_data.get("rent_per_unit", [])),
                    ("Vacancy Rate (%)", avg_cash_flow_data.get("vacancy_rate", [])),
                    ("Rent Growth (%)", avg_cash_flow_data.get("rent_growth_pct", [])),
                    ("Expense Growth (%)", avg_cash_flow_data.get("expense_growth_pct", [])),
                    ("CapEx Growth (%)", avg_cash_flow_data.get("capex_growth_pct", [])),
                    ("Interest Rate (%)", avg_cash_flow_data.get("interest_rates", []))
                ]

                if np.isfinite(exit_cap_rate):
                    cap_rate_row = [np.nan] * (hold_period_actual - 1) + [exit_cap_rate]
                    assumption_rows.append(("Exit Cap Rate (%)", cap_rate_row))

                clean_assumptions = [(label, vals) for label, vals in assumption_rows if isinstance(vals, list) and len(vals) == hold_period_actual]
                if clean_assumptions:
                    assumption_df = pd.DataFrame({label: vals for label, vals in clean_assumptions}).T
                    assumption_df.columns = [f"Yr {i+1}" for i in range(hold_period_actual)]
                    assumption_df.index.name = "Assumption"

                    def format_assumption_value(val, label):
                        if pd.isna(val): return "-"
                        if "Rent ($/Unit/Mo)" in label: return f"${val:,.0f}"
                        elif "Rate (%)" in label: return f"{val:.1%}" if isinstance(val, float) else f"{val:.1f}%"
                        elif "Growth (%)" in label: return f"{val:.1f}%"
                        else: return f"{val:.2f}"

                    formatted_df = assumption_df.apply(lambda row: pd.Series([format_assumption_value(val, row.name) for val in row], index=row.index), axis=1)
                    st.dataframe(formatted_df.style.set_properties(**{"text-align": "right"}).set_table_styles([{"selector": "th", "props": [("text-align", "left")]}]), use_container_width=True)
                else:
                    st.warning("Could not calculate key assumptions by year.")

        with tabs[tab_keys.index("📉 Dynamics")]:
            st.subheader("Key Simulation Dynamics Over Time")
            # Ensure plot_data and its components are available
            years_list_plot = plot_data.get("years", [])
            rent_norm_plot_data = plot_data.get("rent_norm_plot", {})
            vacancy_plot_df = plot_data.get("vacancy_plot_df")
            scatter_data = plot_data.get("scatter_plot", {})

            # --- Rent vs Normal Plot Call ---
            st.markdown("#### Projected Market Rent vs. Fair Value Rent")
            if (rent_norm_plot_data and years_list_plot and
                all(k in rent_norm_plot_data for k in ["market_p05", "market_p50", "market_p95", "normal_p50"]) and
                all(len(rent_norm_plot_data[k]) == len(years_list_plot) for k in rent_norm_plot_data)):
                try:
                    fig_rent_norm = plot_rent_vs_normal( # Function call remains same
                        years=years_list_plot,
                        market_p05=rent_norm_plot_data["market_p05"],
                        market_p50=rent_norm_plot_data["market_p50"],
                        market_p95=rent_norm_plot_data["market_p95"],
                        normal_p50=rent_norm_plot_data["normal_p50"]
                    )
                    st.plotly_chart(fig_rent_norm, use_container_width=True)
                except Exception as e:
                    st.warning(f"Could not plot Projected Market Rent vs Fair Value Rent: {e}")
                    logging.error(f"Rent vs Fair Value plot error: {e}", exc_info=True)
            else:
                st.warning("Rent path data is missing or invalid. Cannot display plot.")
            st.markdown("---")

            # --- Vacancy Distribution Plot ---
            st.markdown("#### Vacancy Rate Distribution Over Time")
            if vacancy_plot_df is not None and not vacancy_plot_df.empty:
                try:
                    fig_vacancy_dist = plot_vacancy_distribution(vacancy_plot_df)
                    st.plotly_chart(fig_vacancy_dist, use_container_width=True)
                except Exception as e:
                    st.warning(f'Error plotting vacancy distribution: {e}')
                    logging.error(f"Vacancy distribution plot error: {e}", exc_info=True)
            else:
                st.warning('Vacancy distribution data not available or format error.')
            st.markdown("---")

            # --- Terminal Growth vs Exit Cap Plot ---
            st.markdown("#### Terminal Rent Growth vs. Exit Cap Rate")
            if scatter_data and scatter_data.get("term_rent_growth_pct") and scatter_data.get("exit_cap_rate_pct"):
                 try:
                     fig_scatter = plot_terminal_growth_vs_exit_cap(scatter_data)
                     st.plotly_chart(fig_scatter, use_container_width=True)
                 except Exception as e:
                     st.warning(f"Could not plot Terminal Growth vs Exit Cap: {e}")
                     logging.error(f"Terminal Growth vs Exit Cap plot error: {e}", exc_info=True)
            else:
                 st.warning("Scatter plot data is missing or invalid.")
            st.markdown("---")

            # --- Simulated Underlying SOFR Rate Distribution Plot (with Forward Curve) ---
            inputs_for_run = st.session_state.get("inputs", {}) # Assuming current inputs reflect the run
            if inputs_for_run.get("is_variable_rate", False):
                st.markdown("#### Simulated Underlying SOFR Rate Distribution")
                st.caption("Shows the median and 5th-95th percentile range of the simulated base SOFR rate (after floor, before spread), compared to the input Forward SOFR Curve.")
                sim_results = processed_results.get("raw_results_for_audit", [])
                fixed_spread = inputs_for_run.get("sofr_spread", 0.0)
                hold_period_plot = len(years_list_plot)
                underlying_sofr_paths = []
                if hold_period_plot > 0:
                    for sim_result in sim_results:
                        effective_rates = sim_result.get("interest_rates", [])
                        if (isinstance(effective_rates, list) and
                            len(effective_rates) == hold_period_plot and
                            all(isinstance(r, (int, float)) and np.isfinite(r) for r in effective_rates)):
                            underlying_path = [r - fixed_spread for r in effective_rates]
                            underlying_sofr_paths.append(underlying_path)
                        else:
                            logging.debug(f"Skipping invalid interest rate path len {len(effective_rates)} vs {hold_period_plot} or non-finite: {effective_rates}")

                forward_rates_plot = []
                if forward_curve_data and years_list_plot:
                     try:
                         max_curve_year = max(forward_curve_data.keys()) if forward_curve_data else 1
                         fallback_rate = forward_curve_data.get(max_curve_year, 0.0)
                         forward_rates_plot = [forward_curve_data.get(yr, fallback_rate) for yr in years_list_plot]
                     except Exception as e:
                         logging.error(f"Error processing forward curve for plotting: {e}")
                         forward_rates_plot = [np.nan] * len(years_list_plot)
                elif years_list_plot:
                     forward_rates_plot = [np.nan] * len(years_list_plot)

                if underlying_sofr_paths:
                    try:
                        fig_sofr = plot_simulated_sofr_distribution(years_list_plot, underlying_sofr_paths, forward_rates_plot)
                        st.plotly_chart(fig_sofr, use_container_width=True)
                    except Exception as e:
                        st.warning(f"Could not generate SOFR distribution plot: {e}")
                        logging.error(f"Error plotting SOFR distribution: {e}", exc_info=True)
                elif hold_period_plot > 0:
                     st.info("No valid simulated interest rate data found for floating rate simulations.")

            # --- ADD Loan Balance Plot ---
            st.markdown("---"); st.markdown("#### Loan Balance & LTV Over Time")
            # Use get_valid_paths imported from utils
            loan_balance_paths = get_valid_paths(sim_results_completed_audit, "loan_balance", inputs_used_for_run.hold_period)
            ltv_paths = get_valid_paths(sim_results_completed_audit, "ltv_estimate", inputs_used_for_run.hold_period)
            if loan_balance_paths and years_list_plot:
                 try:
                      fig_loan = plot_loan_balance_distribution(years_list_plot, loan_balance_paths, ltv_paths)
                      st.plotly_chart(fig_loan, use_container_width=True)
                 except Exception as e:
                      st.warning(f"Could not plot Loan Balance / LTV: {e}")
                      logging.error(f"Loan Balance plot error: {e}", exc_info=True)
            else:
                 st.warning("Loan Balance data not available for plotting.")
            # --- END Loan Balance Plot ---
        with tabs[tab_keys.index("🛡️ Risk")]:
            # Display Risk metrics (logic seems mostly self-contained)
            st.subheader("Risk Profile & Risk-Adjusted Return Metrics")
            # ... (Keep the risk metrics display logic from original script here) ...
            st.markdown("Metrics based on the distribution of **Levered IRR** outcomes from valid simulation runs.")
            if not finite_levered: st.warning("Levered IRR results are not available for risk analysis.")
            else:
                 mean_l_irr_risk = metrics.get("mean_levered_irr", np.nan)
                 median_l_irr_risk = metrics.get("median_levered_irr", np.nan)
                 std_l_irr = risk_metrics.get("Std Dev IRR", np.nan)
                 sharpe = risk_metrics.get("Sharpe Ratio", np.nan)
                 cv = risk_metrics.get("Coefficient of Variation", np.nan)
                 prob_loss = risk_metrics.get("Prob. Loss (IRR < 0%)", np.nan)
                 prob_hurdle = risk_metrics.get("Prob. Below Hurdle", np.nan)
                 var_95 = risk_metrics.get("Value at Risk (VaR 95%)", np.nan)
                 cvar_95 = risk_metrics.get("Cond. VaR (CVaR 95%)", np.nan)
                 hurdle_rate_disp = inputs_used_for_run.hurdle_rate
                 risk_free_disp = inputs_used_for_run.risk_free_rate
                 col1, col2, col3 = st.columns(3)
                 with col1:
                     st.metric("Mean Levered IRR", f"{mean_l_irr_risk:.1%}" if np.isfinite(mean_l_irr_risk) else "N/A")
                     st.metric("Std Dev of IRR", f"{std_l_irr:.1%}" if np.isfinite(std_l_irr) else "N/A", help="Volatility of Levered IRR.")
                     sharpe_label = f"Sharpe Ratio (vs {risk_free_disp:.1%})" if np.isfinite(risk_free_disp) else "Sharpe Ratio"
                     st.metric(sharpe_label, f"{sharpe:.2f}" if np.isfinite(sharpe) else "N/A", help="Risk-adjusted return (higher is better).")
                 with col2:
                     st.metric("Median Levered IRR", f"{median_l_irr_risk:.1%}" if np.isfinite(median_l_irr_risk) else "N/A")
                     st.metric("Value at Risk (VaR 95%)", f"{var_95:.1%}" if np.isfinite(var_95) else "N/A", help="5th percentile outcome.")
                     st.metric("Cond. VaR (CVaR 95%)", f"{cvar_95:.1%}" if np.isfinite(cvar_95) else "N/A", help="Expected IRR in worst 5%.")
                 with col3:
                     st.metric("Prob. Loss (IRR < 0%)", f"{prob_loss:.1%}" if np.isfinite(prob_loss) else "N/A")
                     hurdle_label_short = f"Prob < Hurdle ({hurdle_rate_disp:.0%})" if np.isfinite(hurdle_rate_disp) else "Prob < Hurdle"
                     st.metric(hurdle_label_short, f"{prob_hurdle:.1%}" if np.isfinite(prob_hurdle) else "N/A")
                     st.metric("Coef. of Variation (CV)", f"{cv:.2f}" if np.isfinite(cv) else "N/A", help="Relative risk (lower is better).")
                 # ... (Keep box plot logic) ...
                 st.markdown("---"); st.subheader("Levered IRR Distribution Visualization")
                 failure_threshold = -0.99
                 finite_levered_plot_data = [irr for irr in finite_levered if irr > failure_threshold]
                 num_excluded = len(finite_levered) - len(finite_levered_plot_data)
                 if finite_levered_plot_data:
                     try:
                         fig_box = go.Figure()
                         fig_box.add_trace(go.Box(y=finite_levered_plot_data, name='Levered IRR Dist.', marker_color='rgba(255, 127, 14, 0.7)', boxpoints='outliers', jitter=0.3, pointpos=-1.8, hoverinfo='y'))
                         mean_filtered = np.mean(finite_levered_plot_data)
                         if np.isfinite(mean_filtered): fig_box.add_trace(go.Scatter(x=['Levered IRR Dist.'], y=[mean_filtered], mode='markers', marker=dict(color='red', size=8, symbol='cross-thin'), name='Mean (Filtered)', hoverinfo='skip'))
                         fig_box.update_layout(title="Box Plot of Calculated Levered IRR Outcomes", yaxis_title="Levered IRR", xaxis_title="", template="plotly_white", yaxis_tickformat=".1%", showlegend=False)
                         fig_box.update_xaxes(showticklabels=False)
                         st.plotly_chart(fig_box, use_container_width=True)
                         if num_excluded > 0: st.caption(f"Note: {num_excluded} extreme outlier(s) excluded from plot.")
                     except Exception as e: st.warning(f"Could not generate Levered IRR Box Plot: {e}")
                 else: st.info(f"No Levered IRR data available to display after filtering {num_excluded} outlier(s).")


        with tabs[tab_keys.index("🔍 Audit")]:
            st.subheader("Detailed Audit of Individual Simulation")
            num_completed_simulations = len(sim_results_completed_audit)
            max_sim_sel = max(1, num_completed_simulations)
            if 'selected_sim_index' not in st.session_state:
                st.session_state.selected_sim_index = 0
            current_index = min(max(0, st.session_state.selected_sim_index), max_sim_sel - 1)
            selected_sim_display = st.number_input(
                "Select Simulation # to Audit",
                min_value=1,
                max_value=max_sim_sel,
                value=current_index + 1,
                step=1,
                disabled=(num_completed_simulations == 0),
                key="audit_sim_selector"
            )
            st.session_state.selected_sim_index = selected_sim_display - 1
            selected_sim_index = min(max(0, st.session_state.selected_sim_index), max_sim_sel - 1)
            if num_completed_simulations == 0:
                st.warning("No completed simulations available for audit.")
            else:
                audit_sim = sim_results_completed_audit[selected_sim_index]
                hold_period_audit = len(audit_sim.get("years", []))
                if hold_period_audit <= 0:
                    st.warning(f"Simulation {selected_sim_index+1} incomplete.")
                else:
                    purchase_price = inputs_used_for_run.purchase_price
                    loan_to_cost = inputs_used_for_run.loan_to_cost
                    initial_loan_proceeds = purchase_price * loan_to_cost
                    initial_loan_costs = initial_loan_proceeds * inputs_used_for_run.initial_loan_costs_pct
                    net_initial_loan_proceeds = initial_loan_proceeds - initial_loan_costs
                    initial_equity = purchase_price - net_initial_loan_proceeds

                    refi_year = inputs_used_for_run.refi_year if inputs_used_for_run.enable_refinancing else None
                    # Calculate refinancing proceeds: new loan amount - costs
                    refi_proceeds_net = 0.0
                    if refi_year:
                        noi_at_refi = audit_sim.get("noi", [0.0] * hold_period_audit)[refi_year - 1]
                        refi_cap_rate = inputs_used_for_run.mean_exit_cap_rate / 100.0
                        if noi_at_refi > 0 and refi_cap_rate > FLOAT_ATOL:
                            property_value = noi_at_refi / refi_cap_rate
                            new_loan_amount = property_value * inputs_used_for_run.refi_new_ltv
                            refi_costs = new_loan_amount * inputs_used_for_run.refi_costs_pct_loan
                            refi_proceeds_net = new_loan_amount - refi_costs
                        else:
                            logger.warning(f"Audit Sim {selected_sim_index+1}: Cannot calculate refinancing proceeds: NOI={noi_at_refi}, CapRate={refi_cap_rate}")
                    # Loan payoff at refi is the loan balance at the start of the refi year
                    refi_loan_payoff = audit_sim.get("loan_balance", [0.0] * hold_period_audit)[refi_year - 2] if refi_year and refi_year > 1 else 0.0

                    net_sale_proceeds = audit_sim.get("exit_value_net", 0.0)
                    final_loan_payoff = audit_sim.get("loan_balance", [0.0])[-1] if audit_sim.get("loan_balance") else 0.0

                    # Initialize rows for Net Loan Proceeds and Loan Payoff
                    net_loan_proceeds_row = [net_initial_loan_proceeds] + [0.0] * hold_period_audit
                    loan_payoff_row = [0.0] * (hold_period_audit + 1)

                    if refi_year:
                        net_loan_proceeds_row[refi_year] = refi_proceeds_net
                        loan_payoff_row[refi_year] = -refi_loan_payoff

                    loan_payoff_row[-1] = -final_loan_payoff  # Final loan payoff at sale

                    last_unlev_cf = audit_sim.get("unlevered_cf", [0.0])[-1] if audit_sim.get("unlevered_cf") else 0.0
                    last_lev_cf = audit_sim.get("levered_cf", [0.0])[-1] if audit_sim.get("levered_cf") else 0.0
                    final_unlev_cf = last_unlev_cf + net_sale_proceeds
                    final_lev_cf = last_lev_cf + net_sale_proceeds - final_loan_payoff

                    total_years = hold_period_audit + 1

                    def _pad_list(data_list: Optional[List[Any]], expected_len: int, pad_value: Any = np.nan) -> List[Any]:
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

                    audit_rows = [
                        ("--- Unlevered Cash Flows ---", [np.nan] * total_years),
                        ("Purchase Price", [-purchase_price] + [np.nan] * hold_period_audit),
                        ("--- Operations ---", [np.nan] * total_years),
                        ("Potential Gross Rent (PGR)", _pad_list([np.nan] + audit_sim.get("potential_rent", []), total_years)),
                        ("Less: Vacancy Loss", _pad_list([np.nan] + [-abs(v) for v in audit_sim.get("vacancy_loss", [])], total_years)),
                        ("Effective Gross Rent (EGR)", _pad_list([np.nan] + audit_sim.get("egr", []), total_years)),
                        ("Plus: Other Income", _pad_list([np.nan] + audit_sim.get("other_income", []), total_years)),
                        ("Effective Gross Income (EGI)", _pad_list([np.nan] + audit_sim.get("egi", []), total_years)),
                        ("Less: Operating Expenses", _pad_list([np.nan] + [-abs(e) for e in audit_sim.get("expenses", [])], total_years)),
                        ("Net Operating Income (NOI)", _pad_list([np.nan] + audit_sim.get("noi", []), total_years)),
                        ("Less: Capital Expenditures (CapEx)", _pad_list([np.nan] + [-abs(c) for c in audit_sim.get("capex", [])], total_years)),
                        ("Unlevered Cash Flow (Op.)", _pad_list([np.nan] + audit_sim.get("unlevered_cf", []), total_years)),
                        ("--- Levered Cash Flows ---", [np.nan] * total_years),
                        ("Net Loan Proceeds", net_loan_proceeds_row),
                        ("Loan Payoff", loan_payoff_row),
                        ("Initial Equity", [-initial_equity] + [np.nan] * hold_period_audit),
                        ("--- Debt Service ---", [np.nan] * total_years),
                        ("Interest Paid", _pad_list([np.nan] + [-abs(i) for i in audit_sim.get("interest", [])], total_years)),
                        ("Principal Paid", _pad_list([np.nan] + [-abs(p) for p in audit_sim.get("principal", [])], total_years)),
                        ("Levered Cash Flow (Op.)", _pad_list([np.nan] + audit_sim.get("levered_cf", []), total_years)),
                        ("--- Sale ---", [np.nan] * total_years),
                        ("Net Sale Proceeds", [np.nan] * hold_period_audit + [net_sale_proceeds]),
                        ("--- Cash Flows for IRR ---", [np.nan] * total_years),
                        ("Unlevered Cash Flow (IRR)", _pad_list([-purchase_price] + audit_sim.get("unlevered_cf", [])[:-1] + [final_unlev_cf], total_years)),
                        ("Levered Cash Flow (IRR)", _pad_list([-initial_equity] + audit_sim.get("levered_cf", [])[:-1] + [final_lev_cf], total_years)),
                        ("--- Other Info ---", [np.nan] * total_years),
                        ("End of Year Loan Balance", _pad_list([net_initial_loan_proceeds] + audit_sim.get("loan_balance", []), total_years))
                    ]

                    clean_rows = [
                        (label, vals if isinstance(vals, list) and len(vals) == total_years else [np.nan] * total_years)
                        for label, vals in audit_rows
                    ]

                    audit_df = pd.DataFrame.from_dict(
                        {label: vals for label, vals in clean_rows},
                        orient="index",
                        columns=[f"Year {i}" for i in range(total_years)]
                    )

                    def fmt_audit(val):
                        if pd.isna(val):
                            return "-"
                        try:
                            num_val = float(val)
                            if abs(num_val) >= 1:
                                return f"$({abs(num_val):,.0f})" if num_val < 0 else f"${num_val:,.0f}"
                            elif abs(num_val) > 1e-3:
                                return f"$({abs(num_val):,.2f})" if num_val < 0 else f"${num_val:,.2f}"
                            else:
                                return "$0"
                        except (TypeError, ValueError):
                            return str(val)

                    st.dataframe(
                        audit_df.style.format(fmt_audit, na_rep="-")
                        .set_properties(**{"text-align": "right"})
                        .set_table_styles([{"selector": "th", "props": [("text-align", "left")]}]),
                        use_container_width=True
                    )

                    # Download button
                    st.download_button(
                        label=f"Download Sim {selected_sim_index+1} Data (CSV)",
                        data=audit_df.to_csv().encode('utf-8'),
                        file_name=f"simulation_{selected_sim_index+1}_audit.csv",
                        mime="text/csv",
                        key=f"download_audit_{selected_sim_index}"
                    )

                    # IRR Metrics
                    st.markdown("##### IRR for Selected Simulation")
                    unlevered_irr_val = audit_sim.get('unlevered_irr', np.nan)
                    levered_irr_val = audit_sim.get('levered_irr', np.nan)
                    col1, col2 = st.columns(2)
                    col1.metric("Unlevered IRR", f"{unlevered_irr_val:.1%}" if np.isfinite(unlevered_irr_val) else "N/A")
                    col2.metric("Levered IRR", f"{levered_irr_val:.1%}" if np.isfinite(levered_irr_val) else "N/A")

                    # Key Assumptions Table
                    st.markdown("---")
                    st.subheader("Key Assumptions & Metrics (Simulation #{})".format(selected_sim_index + 1))
                    exit_cap_rate = audit_sim.get("sim_exit_cap_rate", np.nan)
                    assumption_rows = [
                        ("Market Rent ($/Unit/Mo)", audit_sim.get("rent_per_unit", [])),
                        ("Vacancy Rate (%)", audit_sim.get("vacancy_rate", [])),
                        ("Rent Growth (%)", audit_sim.get("rent_growth_pct", [])),
                        ("Expense Growth (%)", audit_sim.get("expense_growth_pct", [])),
                        ("CapEx Growth (%)", audit_sim.get("capex_growth_pct", [])),
                        ("Interest Rate (%)", audit_sim.get("interest_rates", []))
                    ]
                    if np.isfinite(exit_cap_rate):
                        cap_rate_row = [np.nan] * (hold_period_audit - 1) + [exit_cap_rate]
                        assumption_rows.append(("Exit Cap Rate (%)", cap_rate_row))
                    clean_assumptions = [
                        (label, vals) for label, vals in assumption_rows
                        if isinstance(vals, list) and len(vals) == hold_period_audit
                    ]
                    if clean_assumptions:
                        assumption_df = pd.DataFrame({label: vals for label, vals in clean_assumptions}).T
                        assumption_df.columns = [f"Yr {i+1}" for i in range(hold_period_audit)]
                        assumption_df.index.name = "Assumption"
                        def format_assumption_value(val, label):
                            if pd.isna(val):
                                return "-"
                            if "Rent ($/Unit/Mo)" in label:
                                return f"${val:,.0f}"
                            elif "Rate (%)" in label:
                                return f"{val:.1%}" if isinstance(val, float) else f"{val:.1f}%"
                            elif "Growth (%)" in label:
                                return f"{val:.1f}%"
                            else:
                                return f"{val:.2f}"
                        formatted_df = assumption_df.apply(
                            lambda row: pd.Series([format_assumption_value(val, row.name) for val in row], index=row.index),
                            axis=1
                        )
                        st.dataframe(
                            formatted_df.style.set_properties(**{"text-align": "right"})
                            .set_table_styles([{"selector": "th", "props": [("text-align", "left")]}]),
                            use_container_width=True
                        )
                    else:
                        st.warning("Could not calculate key assumptions for this simulation.")
        with tabs[tab_keys.index("🚪 Exit")]:
            # Display Exit analysis plots (logic seems mostly self-contained)
            st.subheader("Exit Analysis")
            # ... (Keep the exit analysis display logic from original script here) ...
            st.caption("Distribution of Net Sale Proceeds & Simulated Exit Cap Rates across all valid runs.")
            if not finite_exit_values or not finite_exit_caps: st.warning("Not enough valid exit results for analysis.")
            else:
                 mean_exit_val = metrics.get('mean_exit_value', np.nan); median_exit_val = metrics.get('median_exit_value', np.nan); p05_exit_val = metrics.get('p05_exit_value', np.nan); p95_exit_val = metrics.get('p95_exit_value', np.nan)
                 mean_exit_cap = metrics.get('mean_exit_cap', np.nan); median_exit_cap = metrics.get('median_exit_cap', np.nan)
                 st.markdown("#### Net Exit Value (After Transaction Costs)"); col1_exit, col2_exit, col3_exit, col4_exit = st.columns(4)
                 col1_exit.metric("Mean", f"${mean_exit_val:,.0f}" if np.isfinite(mean_exit_val) else "N/A"); col2_exit.metric("Median", f"${median_exit_val:,.0f}" if np.isfinite(median_exit_val) else "N/A"); col3_exit.metric("5th Pctl", f"${p05_exit_val:,.0f}" if np.isfinite(p05_exit_val) else "N/A"); col4_exit.metric("95th Pctl", f"${p95_exit_val:,.0f}" if np.isfinite(p95_exit_val) else "N/A")
                 try: # Exit Value Histogram
                     hist_vals, bin_edges = np.histogram(finite_exit_values, bins=30); bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
                     fig_exit_val = go.Figure(data=[go.Bar(x=bin_centers, y=hist_vals, marker_color='mediumseagreen', opacity=0.8)])
                     fig_exit_val.update_layout(title=f"Distribution of Net Exit Values ({len(finite_exit_values)} runs)", xaxis_title="Net Exit Value ($)", yaxis_title="Frequency", bargap=0.1, template="plotly_white"); fig_exit_val.update_xaxes(tickformat="$,.0s")
                     st.plotly_chart(fig_exit_val, use_container_width=True)
                 except Exception as e: logger.error(f"Error plotting exit value distribution: {e}"); st.warning("Could not plot exit value distribution.")
                 st.markdown("---"); st.markdown("#### Simulated Exit Cap Rate"); col1_cap, col2_cap, col3_cap, col4_cap = st.columns(4)
                 p05_cap = np.percentile(finite_exit_caps, 5) if finite_exit_caps else np.nan; p95_cap = np.percentile(finite_exit_caps, 95) if finite_exit_caps else np.nan
                 col1_cap.metric("Mean", f"{mean_exit_cap*100:.2f}%" if np.isfinite(mean_exit_cap) else "N/A"); col2_cap.metric("Median", f"{median_exit_cap*100:.2f}%" if np.isfinite(median_exit_cap) else "N/A"); col3_cap.metric("5th Pctl", f"{p05_cap*100:.2f}%" if np.isfinite(p05_cap) else "N/A"); col4_cap.metric("95th Pctl", f"{p95_cap*100:.2f}%" if np.isfinite(p95_cap) else "N/A")
                 try: # Exit Cap Histogram
                     valid_exit_caps_pct = [x * 100 for x in finite_exit_caps]
                     hist_caps, bin_edges_caps = np.histogram(valid_exit_caps_pct, bins=30); bin_centers_caps = 0.5 * (bin_edges_caps[1:] + bin_edges_caps[:-1])
                     fig_exit_cap = go.Figure(data=[go.Bar(x=bin_centers_caps, y=hist_caps, marker_color='mediumpurple', opacity=0.8)])
                     fig_exit_cap.update_layout(title=f"Distribution of Simulated Exit Cap Rates ({len(finite_exit_caps)} runs)", xaxis_title="Exit Cap Rate (%)", yaxis_title="Frequency", bargap=0.1, template="plotly_white"); fig_exit_cap.update_xaxes(ticksuffix="%")
                     st.plotly_chart(fig_exit_cap, use_container_width=True)
                 except Exception as e: logger.error(f"Error plotting exit cap distribution: {e}"); st.warning("Could not plot exit cap rate distribution.")


        with tabs[tab_keys.index("🔎 Sensitivity")]:
            # Sensitivity Analysis logic
            st.subheader("Sensitivity Analysis")
            # ... (Keep the sensitivity analysis logic from original script here) ...
            # Ensure it uses run_monte_carlo from the core module
            st.markdown("Analyze how key inputs affect the Mean Levered IRR based on the *last run simulation* results' baseline.")
            st.caption("Test multiple parameters. Each parameter will be shown in a separate chart.")
            param_options = { # Define parameter options (internal key, display unit type)
                 "Purchase Price ($)": ("purchase_price", "currency"), "Initial Market Rent ($/Mo)": ("base_rent", "currency"),
                 "Initial Market Rent Premium/Discount to Fair Value (%)": ("market_rent_deviation_pct", "percent_decimal"), "Years to Normalize Rent": ("market_convergence_years", "integer"),
                 "Avg Fair Value Rent Growth (Normal, %)": ("normal_growth_mean", "percent_direct"), "Fair Value Rent Growth Vol (Normal, %)": ("normal_growth_vol", "percent_direct"),
                 "Prob. Normal → Recession (%)": ("transition_normal_to_recession", "percent_decimal"), "Avg Fair Value Rent Growth (Recession, %)": ("recession_growth_mean", "percent_direct"),
                 "Fair Value Rent Growth Vol (Recession, %)": ("recession_growth_vol", "percent_direct"), "Prob. Recession → Normal (%)": ("transition_recession_to_normal", "percent_decimal"),
                 "Starting Vacancy Rate (%)": ("current_vacancy", "percent_decimal"), "Target Long-Term Vacancy (%)": ("stabilized_vacancy", "percent_decimal"),
                 "Vacancy Reversion Speed": ("vacancy_reversion_speed", "decimal_places_2"), "Vacancy Rate Volatility (% pts/Yr)": ("vacancy_volatility", "percent_decimal"),
                 "Initial Annual Other Income ($)": ("mean_other_income", "currency"), "Avg Other Income Growth (%)": ("mean_other_income_growth", "percent_direct"),
                 "Other Income Growth Vol (%)": ("other_income_stddev", "percent_direct"), "Initial Annual OpEx ($)": ("mean_expense", "currency"),
                 "Average OpEx Growth (%)": ("mean_expense_growth", "percent_direct"), "OpEx Growth Volatility (%)": ("expense_stddev", "percent_direct"),
                 "Initial CapEx Reserve ($/Unit/Yr)": ("capex_per_unit_yr", "currency"), "Average CapEx Growth (%)": ("mean_capex_growth", "percent_direct"),
                 "CapEx Growth Volatility (%)": ("capex_stddev", "percent_direct"), "Average Exit Cap Rate (%)": ("mean_exit_cap_rate", "percent_direct"),
                 "Exit Cap Rate Volatility (% pts)": ("exit_cap_rate_stddev", "percent_direct"), "Transaction Cost on Sale (%)": ("transaction_cost_pct", "percent_decimal"),
                 "Exit Cap Adj. for Rent Growth": ("exit_cap_rent_growth_sensitivity", "decimal_places_2"), "Gross Exit Value Floor ($)": ("exit_floor_value", "currency"),
                 "Loan-to-Cost Ratio (%)": ("loan_to_cost", "percent_decimal"), "Fixed Loan Interest Rate (%)": ("interest_rate", "percent_decimal"),
                 "Spread Over SOFR (%)": ("sofr_spread", "percent_decimal"), "SOFR Floor (%)": ("sofr_floor", "percent_decimal"),
                 "Floating Rate Volatility Factor": ("volatility_scalar", "decimal_places_1"), "Cyclical Persistence (Rates)": ("rate_persistence_phi", "decimal_places_2"),
                 "Amortization Period (Years)": ("loan_term_yrs", "integer"), "Corr: Rent & Expense Shocks": ("corr_rent_expense", "decimal_places_2"),
                 "Corr: Rent & Other Income Shocks": ("corr_rent_other_income", "decimal_places_2"), "Corr: Rent & Vacancy Shocks": ("corr_rent_vacancy", "decimal_places_2"),
                 "Risk-Free Rate (%)": ("risk_free_rate", "percent_decimal"), "Hurdle Rate (% IRR)": ("hurdle_rate", "percent_decimal"),
                 "Hold Period (Years)": ("hold_period", "integer"), "Cyclical Persistence (Growth)": ("growth_persistence_phi", "decimal_places_2"),
             }
            if "inputs" not in st.session_state or not st.session_state["inputs"]: st.warning("Inputs not found. Run simulation first.")
            else: # Sensitivity Setup and Execution logic...
                selected_params_display = st.multiselect("Select Parameters to Vary (1–5)", options=list(param_options.keys()), max_selections=5, key="multi_sens_params")
                col_step, col_sims = st.columns(2)
                num_steps = col_step.slider("Number of Steps per Parameter", min_value=3, max_value=11, value=5, step=2, key="sens_steps")
                sens_num_sims = col_sims.number_input("Simulations per Step", min_value=10, max_value=5000, value=200, step=50, key="sens_num_sims")
                st.markdown("---"); st.markdown("###### Define Sensitivity Range")
                range_type = st.radio("Range Definition Method", options=["Relative (%)", "Manual (Min/Max)"], key="sens_range_type", horizontal=True, label_visibility="collapsed")
                param_ranges = {}
                if range_type == "Relative (%)":
                    rel_variation_pct = st.slider(f"Relative Variation (+/- % of baseline)", min_value=1.0, max_value=50.0, value=10.0, step=1.0, key="sens_rel_pct", format="%.0f%%") / 100.0
                    for param_display_name in selected_params_display: param_ranges[param_display_name] = {"type": "relative", "value": rel_variation_pct}
                elif range_type == "Manual (Min/Max)":
                    st.caption("Define the Min/Max range for each selected parameter below:")
                    for param_display_name in selected_params_display: # Manual range input logic...
                        param_key, unit_type = param_options[param_display_name]; base_value = st.session_state['inputs'].get(param_key)
                        # --- Determine format/step ---
                        if unit_type == "percent_decimal": base_disp = base_value * 100.0 if base_value is not None else 0.0; step = 0.1; fmt = "%.1f%%"; min_val_ui = base_disp - 1.0; max_val_ui = base_disp + 1.0
                        elif unit_type == "percent_direct": base_disp = base_value if base_value is not None else 0.0; step = 0.1; fmt = "%.1f%%"; min_val_ui = base_disp - 1.0; max_val_ui = base_disp + 1.0
                        elif unit_type == "currency": base_disp = base_value if base_value is not None else 0.0; step = float(max(1000, abs(base_disp*0.01))); fmt = "$%.0f"; min_val_ui = base_disp * 0.9; max_val_ui = base_disp * 1.1
                        elif unit_type == "integer": base_disp = base_value if base_value is not None else 0; step = 1; fmt = "%d"; min_val_ui = max(0, base_disp - 2); max_val_ui = base_disp + 2
                        elif "decimal_places" in unit_type: base_disp = base_value if base_value is not None else 0.0; places = int(unit_type.split('_')[-1]); step = 1 / (10**places); fmt = f"%.{places}f"; min_val_ui = base_disp - step * 10; max_val_ui = base_disp + step * 10
                        else: base_disp = base_value if base_value is not None else 0.0; step = 0.01; fmt = "%.2f"; min_val_ui = base_disp - 0.1; max_val_ui = base_disp + 0.1
                        # --- End format/step ---
                        st.write(f"**{param_display_name}** (Baseline: {fmt % base_disp if '%' not in fmt else (fmt[:-1] % base_disp) + '%'})")
                        col_min, col_max = st.columns(2)
                        min_r = col_min.number_input("Min Value", value=min_val_ui, step=step, format=fmt, key=f"sens_min_{param_key}", label_visibility="collapsed")
                        max_r = col_max.number_input("Max Value", value=max_val_ui, step=step, format=fmt, key=f"sens_max_{param_key}", label_visibility="collapsed")
                        param_ranges[param_display_name] = {"type": "manual", "min": min_r, "max": max_r}

                st.markdown("---")
                run_button_sens = st.button("Run Sensitivity Analysis", key="run_multi_sensitivity")
                if run_button_sens and selected_params_display: # Sensitivity execution logic...
                    base_inputs_sens = st.session_state['inputs'].copy(); all_sensitivity_results = []
                    baseline_irr = st.session_state.get("processed_results", {}).get("metrics", {}).get("mean_levered_irr", np.nan)
                    for param_display_name in selected_params_display: # Loop through params...
                        param_key, unit_type = param_options[param_display_name]; base_value = base_inputs_sens.get(param_key); range_settings = param_ranges.get(param_display_name)
                        if base_value is None or range_settings is None: st.warning(f"Skipping '{param_display_name}': Missing baseline or range."); continue
                        test_values_internal = []; test_values_display = [] # Determine test values...
                        if range_settings["type"] == "relative":
                            rel_pct = range_settings["value"]; min_val_internal = base_value * (1.0 - rel_pct); max_val_internal = base_value * (1.0 + rel_pct)
                            if np.isclose(base_value, 0): # Handle zero baseline
                                # --- Determine default absolute variation for zero baseline ---
                                if unit_type == "percent_decimal": default_abs_var = 0.01
                                elif unit_type == "percent_direct": default_abs_var = 1.0
                                elif unit_type == "decimal_places_2": default_abs_var = 0.05
                                else: default_abs_var = 0.1
                                # --- End default absolute variation ---
                                min_val_internal = base_value - default_abs_var; max_val_internal = base_value + default_abs_var
                            test_values_internal = np.linspace(min_val_internal, max_val_internal, num_steps)
                        elif range_settings["type"] == "manual":
                            min_r = range_settings["min"]; max_r = range_settings["max"]
                            if unit_type == "percent_decimal" or unit_type == "percent_direct": min_val_internal = float(min_r) / 100.0 if unit_type == "percent_decimal" else float(min_r); max_val_internal = float(max_r) / 100.0 if unit_type == "percent_decimal" else float(max_r)
                            else: min_val_internal = float(min_r); max_val_internal = float(max_r)
                            test_values_internal = np.linspace(min_val_internal, max_val_internal, num_steps)
                        # Determine display values...
                        if unit_type == "percent_decimal": test_values_display = [v * 100.0 for v in test_values_internal]; base_value_display = base_value * 100.0; x_suffix = " (%)"; x_format = "%.1f%%"
                        elif unit_type == "percent_direct": test_values_display = test_values_internal; base_value_display = base_value; x_suffix = " (%)"; x_format = "%.1f%%"
                        elif unit_type == "currency": test_values_display = test_values_internal; base_value_display = base_value; x_suffix = " ($)"; x_format = "$%.0f"
                        elif unit_type == "integer": test_values_display = np.round(test_values_internal).astype(int); base_value_display = int(round(base_value)); x_suffix = ""; x_format = "%d"
                        elif "decimal_places" in unit_type: test_values_display = test_values_internal; base_value_display = base_value; places = int(unit_type.split('_')[-1]); x_suffix = ""; x_format = f"%.{places}f"
                        else: test_values_display = test_values_internal; base_value_display = base_value; x_suffix = ""; x_format = "%.2f"
                        irrs_for_param = [] # Run simulations for this param...
                        status_msg = f"Running sensitivity for {param_display_name}...";
                        with st.status(status_msg, expanded=True) as status:
                            for i, internal_val in enumerate(test_values_internal):
                                display_val_str = x_format % test_values_display[i];
                                if '%' in display_val_str: display_val_str = display_val_str.replace('%','') + '%'
                                st.write(f"Step {i+1}/{num_steps}: Testing {param_key} = {display_val_str}...")
                                temp_inputs_dict = base_inputs_sens.copy(); temp_inputs_dict[param_key] = internal_val
                                try:
                                    temp_sim_inputs = SimulationInputs(**temp_inputs_dict) # Create object for run
                                    result = run_monte_carlo(temp_sim_inputs, sens_num_sims, forward_curve_data, std_dev_curve_data) # Call core function
                                    irr = result["metrics"].get("mean_levered_irr", np.nan)
                                    irrs_for_param.append(irr)
                                    st.write(f"-> Mean Levered IRR: {irr:.1%}" if np.isfinite(irr) else "-> Mean Levered IRR: N/A")
                                except Exception as e: irrs_for_param.append(np.nan); st.write(f"-> Error running simulation: {e}"); logger.error(f"Sensitivity failed for {param_key}={internal_val}: {e}")
                            status.update(label=f"Sensitivity analysis for {param_display_name} complete!", state="complete", expanded=False)
                        # Store results and plot...
                        for disp_val, irr in zip(test_values_display, irrs_for_param): all_sensitivity_results.append({"Parameter": param_display_name, "Value_Display": disp_val, "Mean Levered IRR": irr * 100.0 if np.isfinite(irr) else None})
                        fig_sens = go.Figure(); valid_indices = [i for i, irr in enumerate(irrs_for_param) if np.isfinite(irr)]; plot_x = [test_values_display[i] for i in valid_indices]; plot_y = [irrs_for_param[i] * 100 for i in valid_indices]
                        if plot_x and plot_y:
                            fig_sens.add_trace(go.Scatter(x=plot_x, y=plot_y, mode="lines+markers", name=param_display_name))
                            if np.isfinite(baseline_irr): # Add baseline marker
                                closest_x_to_baseline = min(test_values_display, key=lambda x:abs(x-base_value_display))
                                fig_sens.add_trace(go.Scatter(x=[base_value_display], y=[baseline_irr * 100], mode="markers", marker=dict(color="red", size=10, symbol="x"), name="Baseline"))
                            baseline_text_fmt = x_format % base_value_display;
                            if '%' in baseline_text_fmt: baseline_text_fmt = baseline_text_fmt.replace('%','') + '%'
                            baseline_text = f"Baseline: {baseline_text_fmt}"
                            fig_sens.add_vline(x=base_value_display, line_dash="dash", line_color="rgba(255,0,0,0.5)", annotation_text=baseline_text, annotation_position="bottom right")
                            fig_sens.update_layout(title=f"Mean Levered IRR Sensitivity to {param_display_name}", xaxis_title=f"{param_display_name}{x_suffix}", yaxis_title="Mean Levered IRR (%)", yaxis_tickformat=".1f", hovermode="x unified", template="plotly_white", showlegend=True, legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
                            if unit_type == "currency": fig_sens.update_xaxes(tickformat="$,.0f")
                            elif '%' in x_suffix: fig_sens.update_xaxes(ticksuffix="%")
                            st.plotly_chart(fig_sens, use_container_width=True)
                        else: st.warning(f"No valid IRR results for {param_display_name}.")
                        st.markdown("---")
                    # Download button for all results...
                    if all_sensitivity_results:
                        csv_df = pd.DataFrame(all_sensitivity_results); csv_bytes = csv_df.to_csv(index=False).encode('utf-8')
                        st.download_button(label="📥 Download Full Sensitivity Results as CSV", data=csv_bytes, file_name="sensitivity_analysis_results.csv", mime="text/csv")


        with tabs[tab_keys.index("🗂️ Scenarios")]:
            # Scenario comparison logic (seems mostly self-contained)
            st.subheader("Scenario Management & Comparison")
            # ... (Keep scenario saving/loading/comparison logic from original script here) ...
            # Ensure it uses st.session_state['saved_scenarios']
            st.markdown("### Save Current Scenario Results"); scenario_name = st.text_input("Scenario Name", key="save_scenario_name_mgmt", help="Enter a unique name for this scenario run."); scenario_desc = st.text_area("Optional Notes / Tags", key="save_scenario_desc_mgmt", help="Add notes about assumptions or purpose.")
            if st.button("📎 Save Current Scenario & Results", key="save_button_mgmt", use_container_width=True): # Save logic...
                if scenario_name.strip():
                     saved_result = st.session_state.get("processed_results")
                     if saved_result and isinstance(saved_result, dict) and "error" not in saved_result:
                         inputs_copy = st.session_state["inputs"].copy(); results_copy = saved_result.copy()
                         from datetime import datetime # Local import if needed
                         st.session_state["saved_scenarios"][scenario_name] = {"inputs": inputs_copy, "results": results_copy, "description": scenario_desc, "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
                         st.success(f"Saved scenario '{scenario_name}'.")
                     else: st.warning("No valid results to save.")
                else: st.warning("Please enter a scenario name.")
            st.markdown("---"); st.markdown("### Saved Scenarios") # Display saved scenarios...
            if not st.session_state.get("saved_scenarios"): st.info("No scenarios saved yet.")
            else:
                 sorted_scenario_keys = sorted(st.session_state["saved_scenarios"].keys())
                 for name in sorted_scenario_keys:
                     if name in st.session_state["saved_scenarios"]: # Check existence
                         details = st.session_state["saved_scenarios"][name]
                         with st.expander(f"{name} — Saved: {details.get('timestamp', 'N/A')}"): # Expander logic...
                             st.caption(f"Notes: {details.get('description', '_No description provided._')}"); col1, col2, col3 = st.columns([1, 1, 2])
                             if col1.button("🔄 Load Inputs", key=f"load_{name}", help="Load inputs from this scenario into the sidebar."): # Load button...
                                 st.session_state["inputs"] = details["inputs"].copy(); st.session_state["processed_results"] = None; st.success(f"Inputs from '{name}' loaded."); time.sleep(1); st.rerun()
                             if col2.button("🗑️ Delete", key=f"delete_{name}", help="Delete this saved scenario."): # Delete button...
                                 del st.session_state["saved_scenarios"][name]; st.rerun()
                             new_name = col3.text_input(f"Rename '{name}' to:", value=name, key=f"rename_{name}", label_visibility="collapsed"); rename_button_key = f"rename_btn_{name}" # Rename logic...
                             if new_name != name and new_name.strip():
                                 if new_name in st.session_state["saved_scenarios"]: col3.warning("Name exists.")
                                 elif col3.button("✏️ Rename", key=rename_button_key): st.session_state["saved_scenarios"][new_name] = st.session_state["saved_scenarios"].pop(name); st.rerun()
            st.markdown("---"); st.markdown("### Compare Two Saved Scenarios") # Comparison logic...
            if len(st.session_state.get("saved_scenarios", {})) < 2: st.info("Save at least two scenarios.")
            else: # Scenario comparison UI and logic...
                 scenario_keys = sorted(list(st.session_state["saved_scenarios"].keys())); col_a, col_b = st.columns(2)
                 scenario_a = col_a.selectbox("Select Scenario A", options=scenario_keys, index=0, key="comp_a")
                 default_b_index = 1 if len(scenario_keys) > 1 else 0;
                 if scenario_a == scenario_keys[default_b_index] and len(scenario_keys) > 1: default_b_index = 0
                 scenario_b = col_b.selectbox("Select Scenario B", options=scenario_keys, index=default_b_index, key="comp_b")
                 if scenario_a and scenario_b and scenario_a != scenario_b: # Comparison display logic...
                      if scenario_a not in st.session_state["saved_scenarios"] or scenario_b not in st.session_state["saved_scenarios"]: st.warning("Scenario not found.")
                      else: # Get data and display comparison tables/plots...
                          data_a = st.session_state["saved_scenarios"][scenario_a]; data_b = st.session_state["saved_scenarios"][scenario_b]; result_a = data_a.get("results", {}); result_b = data_b.get("results", {}); inputs_a = data_a.get("inputs", {}); inputs_b = data_b.get("inputs", {})
                          # ... (Keep comparison metric table logic) ...
                          st.markdown("#### 🔍 Scenario Comparison: Key Metrics")
                          def get_metric_and_delta(metric_key_path, res_a, res_b): # Helper defined inside tab scope
                              val_a = res_a; val_b = res_b
                              try:
                                  for key in metric_key_path: val_a = val_a.get(key, {})
                                  for key in metric_key_path: val_b = val_b.get(key, {})
                                  val_a = val_a if isinstance(val_a, (int, float, np.number)) else None; val_b = val_b if isinstance(val_b, (int, float, np.number)) else None
                                  delta = None;
                                  if val_a is not None and val_b is not None and np.isfinite(val_a) and np.isfinite(val_b): delta = val_b - val_a
                                  return val_a, val_b, delta
                              except Exception as e: logger.error(f"Error get_metric_delta {metric_key_path}: {e}"); return None, None, None
                          metrics_to_show_comp = [ # Define metrics to compare
                              ("Mean Levered IRR", ("metrics", "mean_levered_irr"), lambda x: f"{x:.1%}" if x is not None and np.isfinite(x) else "N/A", lambda d: f"{d*100:.1f} pts" if d is not None else ""),
                              ("Median Levered IRR", ("metrics", "median_levered_irr"), lambda x: f"{x:.1%}" if x is not None and np.isfinite(x) else "N/A", lambda d: f"{d*100:.1f} pts" if d is not None else ""),
                              ("Mean Unlevered IRR", ("metrics", "mean_unlevered_irr"), lambda x: f"{x:.1%}" if x is not None and np.isfinite(x) else "N/A", lambda d: f"{d*100:.1f} pts" if d is not None else ""),
                              ("Median Unlevered IRR", ("metrics", "median_unlevered_irr"), lambda x: f"{x:.1%}" if x is not None and np.isfinite(x) else "N/A", lambda d: f"{d*100:.1f} pts" if d is not None else ""),
                              ("Mean Net Exit Value", ("metrics", "mean_exit_value"), lambda x: f"${x:,.0f}" if x is not None and np.isfinite(x) else "N/A", lambda d: f"${d:,.0f}" if d is not None else ""),
                              ("Mean Exit Cap Rate", ("metrics", "mean_exit_cap"), lambda x: f"{x*100:.2f}%" if x is not None and np.isfinite(x) else "N/A", lambda d: f"{d*100:.2f} pts" if d is not None else ""), # Assuming stored as decimal
                              ("Std Dev of IRR", ("risk_metrics", "Std Dev IRR"), lambda x: f"{x:.1%}" if x is not None and np.isfinite(x) else "N/A", lambda d: f"{d*100:.1f} pts" if d is not None else ""),
                              ("Sharpe Ratio", ("risk_metrics", "Sharpe Ratio"), lambda x: f"{x:.2f}" if x is not None and np.isfinite(x) else "N/A", lambda d: f"{d:.2f}" if d is not None else ""),
                              ("Value at Risk (95%)", ("risk_metrics", "Value at Risk (VaR 95%)"), lambda x: f"{x:.1%}" if x is not None and np.isfinite(x) else "N/A", lambda d: f"{d*100:.1f} pts" if d is not None else ""),
                              ("Cond. VaR (95%)", ("risk_metrics", "Cond. VaR (CVaR 95%)"), lambda x: f"{x:.1%}" if x is not None and np.isfinite(x) else "N/A", lambda d: f"{d*100:.1f} pts" if d is not None else ""),
                              ("Prob. Loss (IRR < 0%)", ("risk_metrics", "Prob. Loss (IRR < 0%)"), lambda x: f"{x:.1%}" if x is not None and np.isfinite(x) else "N/A", lambda d: f"{d*100:.1f} pts" if d is not None else ""),
                              ("Prob. Below Hurdle", ("risk_metrics", "Prob. Below Hurdle"), lambda x: f"{x:.1%}" if x is not None and np.isfinite(x) else "N/A", lambda d: f"{d*100:.1f} pts" if d is not None else ""),
                              ("Coefficient of Variation", ("risk_metrics", "Coefficient of Variation"), lambda x: f"{x:.2f}" if x is not None and np.isfinite(x) else "N/A", lambda d: f"{d:.2f}" if d is not None else "")
                          ]
                          comparison_data_metrics = []
                          for label, key_path, fmt_func, delta_fmt_func in metrics_to_show_comp:
                              val_a, val_b, delta = get_metric_and_delta(key_path, result_a, result_b)
                              row = {"Metric": label, scenario_a: fmt_func(val_a), scenario_b: fmt_func(val_b), "Difference (B - A)": delta_fmt_func(delta) if delta_fmt_func else ""}
                              comparison_data_metrics.append(row)
                          st.dataframe(pd.DataFrame(comparison_data_metrics).set_index("Metric"), use_container_width=True)
                          # ... (Keep assumption comparison logic) ...
                          st.markdown("---"); st.markdown("#### 📋 Assumptions Comparison")
                          def format_value_compare(k, v): # Helper defined inside tab scope
                              if v is None: return "N/A"
                              dollar_keys = {"purchase_price", "mean_other_income", "mean_expense", "exit_floor_value", "base_rent", "capex_per_unit_yr"}; percent_decimal_keys = {"market_rent_deviation_pct", "transition_normal_to_recession", "transition_recession_to_normal", "current_vacancy", "stabilized_vacancy", "vacancy_volatility", "loan_to_cost", "interest_rate", "transaction_cost_pct", "risk_free_rate", "hurdle_rate", "sofr_spread", "sofr_floor"}; percent_direct_keys = {"normal_growth_mean", "normal_growth_vol", "recession_growth_mean", "recession_growth_vol", "mean_other_income_growth", "other_income_stddev", "mean_expense_growth", "expense_stddev", "mean_capex_growth", "capex_stddev", "mean_exit_cap_rate", "exit_cap_rate_stddev"}; float_2dp_keys = {"vacancy_reversion_speed", "growth_persistence_phi", "corr_rent_expense", "corr_rent_other_income", "corr_rent_vacancy", "exit_cap_rent_growth_sensitivity", "rate_persistence_phi"}; float_1dp_keys = {"volatility_scalar"}; int_keys = {"num_units", "market_convergence_years", "loan_term_yrs", "hold_period", "num_simulations"}
                              try:
                                  if k in dollar_keys: return f"${float(v):,.0f}"
                                  elif k in percent_decimal_keys: return f"{float(v)*100:.1f}%"
                                  elif k in percent_direct_keys: return f"{float(v):.1f}%"
                                  elif k in float_2dp_keys: return f"{float(v):.2f}"
                                  elif k in float_1dp_keys: return f"{float(v):.1f}"
                                  elif k in int_keys: return f"{int(v):d}"
                                  elif k == "loan_type": return str(v)
                                  elif k == "use_correlations": return "Yes" if v else "No"
                                  elif k == "is_variable_rate": return "Floating" if v else "Fixed"
                                  elif isinstance(v, (int, float, np.number)): return f"{v:.2f}"
                                  else: return str(v)
                              except (ValueError, TypeError): return str(v)
                          input_labels = {k: k.replace("_", " ").title() for k in SimulationInputs.__annotations__ if not k.startswith('_') and not isinstance(getattr(SimulationInputs, k, None), property)}; differences_summary = []
                          for k in sorted(input_labels.keys()):
                              if k in inputs_a or k in inputs_b:
                                  val_a = inputs_a.get(k); val_b = inputs_b.get(k); fmt_a = format_value_compare(k, val_a); fmt_b = format_value_compare(k, val_b)
                                  if fmt_a != fmt_b: differences_summary.append(f"- **{input_labels[k]}**: {scenario_a}=`{fmt_a}`, {scenario_b}=`{fmt_b}`")
                          if differences_summary: st.markdown("**Key Assumption Differences:**"); st.markdown("\n".join(differences_summary)); st.markdown("---")
                          else: st.info("Inputs appear identical.")
                          rows_assumptions = []
                          for k in sorted(input_labels.keys()):
                              if k in inputs_a or k in inputs_b:
                                  val_a = inputs_a.get(k); val_b = inputs_b.get(k); fmt_a = format_value_compare(k, val_a); fmt_b = format_value_compare(k, val_b); is_different = fmt_a != fmt_b
                                  row = {"Assumption": input_labels[k], scenario_a: fmt_a, scenario_b: fmt_b, "IsDifferent": is_different}; rows_assumptions.append(row)
                          df_compare_assumptions = pd.DataFrame(rows_assumptions).set_index("Assumption")
                          show_only_diff = st.checkbox("Show only differing assumptions", key="scenario_diff_check")
                          df_display_assumptions = df_compare_assumptions[df_compare_assumptions["IsDifferent"]].drop(columns=["IsDifferent"]) if show_only_diff else df_compare_assumptions.drop(columns=["IsDifferent"])
                          if not df_display_assumptions.empty: st.dataframe(df_display_assumptions, use_container_width=True)
                          elif show_only_diff: st.info("No differing assumptions found.")
                          # ... (Updated IRR comparison plot logic) ...
                          st.markdown("#### Levered IRR Distributions")
                          irrs_a = result_a.get("finite_levered_irrs", [])
                          irrs_b = result_b.get("finite_levered_irrs", [])
                          if not irrs_a and not irrs_b: st.warning("No valid Levered IRR data for comparison plot.")
                          else:
                              combined_irrs = [irr for irr in irrs_a if np.isfinite(irr)] + [irr for irr in irrs_b if np.isfinite(irr)]
                              if not combined_irrs: st.warning("No finite Levered IRR data found to plot.")
                              else:
                                   global_min = np.min(combined_irrs); global_max = np.max(combined_irrs)
                                   padding = (global_max - global_min) * 0.05
                                   if np.isclose(global_max, global_min): padding = 0.05
                                   x_start = global_min - padding
                                   x_end = global_max + padding
                                   num_bins = 30
                                   bin_size = (x_end - x_start) / num_bins if not np.isclose(x_end, x_start) else 0.1
                                   fig_comp = make_subplots(
                                       rows=2, cols=1,
                                       shared_xaxes=True, # Share the x-axis
                                       vertical_spacing=0.05, # Reduce space between plots
                                       subplot_titles=(f"Scenario: {scenario_a}", f"Scenario: {scenario_b}")
                                   )
                                   if bin_size > FLOAT_ATOL:
                                       fig_comp.add_trace(go.Histogram(
                                           x=irrs_a, name=scenario_a,
                                           marker_color='rgba(55, 128, 191, 0.7)', # Blue
                                           xbins=dict(start=x_start, end=x_end, size=bin_size),
                                           showlegend=False),
                                           row=1, col=1 # Position: Row 1, Col 1
                                       )
                                       fig_comp.add_trace(go.Histogram(
                                           x=irrs_b, name=scenario_b,
                                           marker_color='rgba(255, 127, 14, 0.7)', # Orange
                                           xbins=dict(start=x_start, end=x_end, size=bin_size),
                                           showlegend=False),
                                           row=2, col=1 # Position: Row 2, Col 1
                                       )
                                       fig_comp.update_layout(
                                           title_text="Levered IRR Distributions (Comparison)",
                                           height=600, template="plotly_white", bargap=0.1,
                                           margin=dict(t=60, b=10, l=10, r=10)
                                       )
                                       fig_comp.update_xaxes(tickformat=".1%", title_text="Levered IRR", row=2, col=1)
                                       fig_comp.update_yaxes(title_text="Frequency", row=1, col=1)
                                       fig_comp.update_yaxes(title_text="Frequency", row=2, col=1)
                                       st.plotly_chart(fig_comp, use_container_width=True)
                                   else:
                                       st.warning("Cannot generate comparison histogram: IRR range too narrow.")
                 elif scenario_a == scenario_b: st.warning("Please select two different scenarios to compare.")

        # --- <<< CORRECTED INDENTATION >>> ---
        with tabs[tab_keys.index("ℹ️ Guide")]:
            st.markdown("""
                ## Overview & Instructions

                Welcome to **PropSaber**, a next-generation real estate simulation model built for serious multifamily investors.

                PropSaber replaces oversimplified spreadsheets with a dynamic, scenario-based engine—giving you insights into how deals might *actually* perform, not just how they look on paper.

                Is it complicated? Yes and no.

                Under the hood, it’s powered by a sophisticated simulation engine using Monte Carlo methods, stochastic processes, and regime switching. But on the surface, it’s been designed for **practitioners, by practitioners**, with one clear goal: **help you make better investment decisions**.

                Whether you’re underwriting a stabilized asset, testing downside risk, or trying to impress an investment committee—PropSaber gives you a richer view of returns, risks, and variability, with tools that feel intuitive and grounded in how investors actually think.

                ### Quick Start Guide

                1.  **Set Inputs**: Use the sidebar on the left. Click `>` to expand sections. Adjust values with the **slider** for quick changes or **type exact numbers** in the box for precision. Defaults are a solid starting point.
                    * Input fields marked with `($)` expect dollar amounts.
                    * Fields marked with `(%)` expect percentages (e.g., enter `5.0` for 5%).
                    * **Financing:** Choose "Fixed" or "Floating" rate types. *Note:* The "Floating" option requires the `Pensford_Forward_Curve.csv` file to be present.
                2.  **Run Simulation**: Click the **🚀 Run Simulation** button at the top of the sidebar to generate results based on your inputs.
                3.  **Explore Results**: Check the tabs (e.g., Summary, IRR, Pro-Forma, Dynamics, Risk) to see outcomes and risks.
                4.  **Iterate**: Tweak inputs and rerun to compare scenarios. The model is designed for rapid iteration.
                5.  **Save/Load/Compare**: Use **💾 Scenario Files** in the sidebar or the **🗂️ Scenarios** tab to save, load, or compare different input sets and their results. After loading, click **Run Simulation** again.

                ### Why Use This Tool?

                Most models give you a single outcome. PropSaber shows you the entire range—and how likely each scenario is.

                - **Monte Carlo Simulation**: Generates thousands of future scenarios based on your rent, expense, vacancy, and exit assumptions.
                - **Realistic Dynamics**: Rent, OpEx, and vacancy evolve over time using **Geometric Brownian Motion** and **mean-reverting** processes, not flat lines. Interest rates (if floating) follow a simulated path based on the forward curve, volatility, and persistence.
                - **Market Regime Switching**: Simulates transitions between “Normal” and “Recession” market states using your probability assumptions in the "Rent" section, affecting rent growth.
                - **Correlations**: Optionally link random shocks between rent, expenses, and vacancy via the "Correlation" section.
                - **Sophisticated Debt Modeling**: Accurately model **Fixed** or **Floating** rate debt, including Interest Only or Amortizing options (Fixed only), realistic floating rate mechanics, and refinancing scenarios (see below).
                - **Risk Metrics That Matter**: Outputs include Sharpe ratio, downside probability (Prob. Loss, Prob. Below Hurdle), Value-at-Risk (VaR), Conditional VaR (CVaR), and Coefficient of Variation – all accessible on the "Risk" tab.

                ### Key Features in the Financing Section

                The **Financing** section lets you model debt with real-world flexibility, including the ability to simulate a refinancing event:

                * **Loan-to-Cost Ratio**: Sets the initial loan amount as a percentage of the purchase price.
                * **Rate Type**:
                    * **Fixed**: Uses the specified "Fixed Loan Interest Rate". You can also choose:
                        * **Loan Type**: "Interest Only" (no principal paid until sale) or "Amortizing" (principal paid down over the "Amortization Period").
                    * **Floating**: Simulates a variable rate based on several factors. *Floating rate loans are currently modeled as Interest Only.*
                        * **Forward SOFR Curve**: The starting point for each year's rate comes from the `Pensford_Forward_Curve.csv` file.
                        * **Interest Rate Volatility**: Adds random annual shocks (normally distributed noise) to the forward SOFR rate.
                        * **Rate Persistence (φ)**: Smooths the `SOFR + shock` component over time using an AR(1) process. A value near 0 means little smoothing; a value near 1 means high persistence from year to year.
                        * **SOFR Floor**: After applying volatility and persistence to the base SOFR rate, the model compares this value to the floor. It takes the **higher** of the two. The rate component (before spread) will not drop below this floor. Enter as a percentage (e.g., `1.0` for 1%).
                        * **Spread Over SOFR**: This fixed spread is added **last**, after the floor has been applied, to determine the final `effective_rate` used for calculating interest payments.
                * **Refinancing Options**: Enable refinancing to model a loan reset during the hold period, reflecting real-world strategies to capitalize on property value appreciation or rate changes:
                    * **Enable Refinancing**: Check this box to activate refinancing. If unchecked, the original loan terms persist throughout the hold period.
                    * **Refinancing Year**: Specify the year (e.g., Year 3) when the refinance occurs. The new loan terms apply starting at the beginning of this year.
                    * **New Loan-to-Value (LTV) Ratio (%)**: Set the target LTV for the new loan, based on the estimated property value in the refinancing year (calculated using NOI and the mean exit cap rate as a proxy).
                    * **Refinancing Costs (% of Loan)**: Enter the costs associated with refinancing (e.g., 1% of the new loan amount) as a percentage. These costs are deducted from the net cash proceeds or added to the loan balance.
                    * **New Amortization Period (Years)**: Define the amortization period for the new loan (e.g., 30 years). The refinanced loan is modeled as amortizing, not interest-only.
                    * **Fixed Rate Spread to SOFR (%)**: Specify the spread added to the SOFR rate in the refinancing year to determine the new fixed interest rate. The model uses the forward SOFR curve for that year as the base rate.
                    * **Impact on Cash Flows**: Refinancing can generate cash proceeds (if the new loan exceeds the existing balance minus costs) or require cash injection (if the new loan is smaller). These cash flows are reflected in the Levered Cash Flow in the refinancing year, impacting IRR calculations.

                ### Exploring Results

                - **Summary Tab**: Shows key KPIs and an initial snapshot based on Year 0 inputs and estimated Year 1 debt service.
                - **IRR Tab**: Displays distributions of Unlevered and Levered IRR outcomes, reflecting any refinancing cash flows.
                - **Pro-Forma Tab**: Presents the average annual cash flows across all simulations, including income, expenses, CapEx, debt service (interest and principal reflecting fixed/floating rates and refinancing), and sale proceeds.
                - **Dynamics Tab**: Visualizes simulation behavior over time, including:
                    * Rent path distribution vs. underlying fair value rent.
                    * Vacancy rate distribution per year.
                    * Relationship between terminal rent growth and exit cap rates.
                    * *(If Floating Rate)* The distribution of the simulated underlying SOFR rate (SOFR + Volatility + Persistence, *before* Spread) compared to the input Forward SOFR Curve.
                    * Loan balance and LTV over time, showing the impact of refinancing on debt levels.
                - **Risk Tab**: Provides detailed risk metrics based on the Levered IRR distribution, accounting for refinancing variability.
                - **Audit Tab**: Allows detailed inspection of the cash flows and metrics for any single simulation run, including refinancing proceeds or costs.
                - **Exit Tab**: Shows distributions for the Net Exit Value and the simulated Exit Cap Rate.
                - **Sensitivity Tab**: Run sensitivity analyses on key inputs, including refinancing parameters, to see their impact on Mean Levered IRR.
                - **Scenarios Tab**: Save named snapshots of your inputs *and results*, load previous scenarios, and compare two scenarios side-by-side, including differences in refinancing strategies.

                ---

                *Disclaimer: This is a simulation tool. Results are illustrative and depend heavily on input assumptions. Not financial advice.*
                """)


# --- Entry Point Check ---
if __name__ == "__main__":
    main() # Ensure standard 4-space indentation
