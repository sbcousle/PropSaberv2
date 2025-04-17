"""
Main Streamlit application entry point for PropSaber.
Orchestrates UI setup (sidebar, tabs), input handling,
simulation execution, and visualization display by calling
functions from the propsaber package modules.
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
        plot_loan_balance_distribution, plot_multiple_distribution
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
        render_sidebar_inputs(
            initial_inputs=SimulationInputs(**st.session_state["inputs"]),
            forward_curve=forward_curve_data,
            std_dev_curve=std_dev_curve_data
        )
        st.markdown("---")
        st.subheader("💾 Scenario Files")
        save_name = st.text_input("New Scenario File Name", value="my_scenario", key="save_scenario_name_sidebar", help="Enter name to save CURRENT sidebar inputs to a file.")
        if st.button("Save Inputs to File", key="save_button_sidebar"):
            if save_name:
                save_scenario(st.session_state["inputs"], filename=f"{save_name}.json")
            else:
                st.warning("Please enter a file name.")
        st.markdown("---")
        try:
            scenario_files = list_saved_scenarios()
        except Exception as e:
            st.error(f"Error reading scenario directory: {e}")
            scenario_files = []
        if scenario_files:
            selected_file_load = st.selectbox("Load or Delete Scenario File", options=[""] + scenario_files, index=0, key="scenario_selector_load_sidebar", help="Select a file to load its inputs into the sidebar or delete it.")
            if selected_file_load:
                col_load, col_delete_init = st.columns(2)
                with col_load:
                    if st.button("✅ Load Inputs", key=f"load_button_{selected_file_load}_sidebar"):
                        loaded_inputs_dict = load_scenario(filename=selected_file_load)
                        if loaded_inputs_dict:
                            st.session_state["inputs"] = loaded_inputs_dict
                            st.session_state["processed_results"] = None
                            st.success(f"Loaded '{selected_file_load}'.")
                            time.sleep(1.5)
                            st.rerun()
                with col_delete_init:
                    if st.button("🗑️ Delete File", key=f"delete_init_{selected_file_load}_sidebar"):
                        st.session_state.confirming_delete = selected_file_load
                        st.rerun()
                if st.session_state.get("confirming_delete") == selected_file_load:
                    st.warning(f"Confirm delete: **{selected_file_load}**?")
                    col_confirm, col_cancel = st.columns(2)
                    with col_confirm:
                        if st.button("✔️ Yes, Delete", key=f"delete_confirm_{selected_file_load}_sidebar"):
                            try:
                                from propsaber.core.constants import SCENARIO_DIR
                                (SCENARIO_DIR / selected_file_load).unlink()
                                st.success(f"Deleted: {selected_file_load}")
                                st.session_state.confirming_delete = None
                                time.sleep(1)
                                st.rerun()
                            except Exception as e:
                                st.error(f"Error deleting file: {e}")
                                st.session_state.confirming_delete = None
                    with col_cancel:
                        if st.button("❌ Cancel", key=f"delete_cancel_{selected_file_load}_sidebar"):
                            st.session_state.confirming_delete = None
                            st.rerun()
        else:
            st.info("No saved scenario files found.")

    # --- Input Processing & Validation ---
    processed_inputs_dict = {}
    inputs_valid_for_processing = True
    default_inputs_instance = SimulationInputs()
    percentage_decimal_keys = {"market_rent_deviation_pct", "transition_normal_to_recession", "transition_recession_to_normal", "current_vacancy", "stabilized_vacancy", "vacancy_volatility", "loan_to_cost", "interest_rate", "transaction_cost_pct", "risk_free_rate", "hurdle_rate", "sofr_spread", "sofr_floor", "initial_loan_costs_pct", "refi_new_ltv", "refi_costs_pct_loan"}
    percentage_direct_keys = {"normal_growth_mean", "normal_growth_vol", "recession_growth_mean", "recession_growth_vol", "mean_other_income_growth", "other_income_stddev", "mean_expense_growth", "expense_stddev", "mean_capex_growth", "capex_stddev", "mean_exit_cap_rate", "exit_cap_rate_stddev", "refi_fixed_rate_spread_to_sofr"}
    all_input_keys = {k for k in SimulationInputs.__annotations__ if not k.startswith('_') and not isinstance(getattr(SimulationInputs, k, None), property)}

    for key in all_input_keys:
        widget_key = f"input_{key}"
        default_value = getattr(default_inputs_instance, key, None)
        default_type = type(default_value) if default_value is not None else None

        if widget_key in st.session_state:
            ui_value = st.session_state[widget_key]
        else:
            ui_value = st.session_state.get("inputs", {}).get(key, default_value)

        if key == 'is_variable_rate':
            processed_inputs_dict[key] = (st.session_state.get("input_rate_type", "Fixed") == "Floating")
            continue
        if key == 'loan_type':
            is_float = processed_inputs_dict.get('is_variable_rate', (st.session_state.get("input_rate_type", "Fixed") == "Floating"))
            if is_float:
                processed_inputs_dict[key] = LOAN_TYPE_IO
            else:
                processed_inputs_dict[key] = st.session_state.get("input_loan_type", default_inputs_instance.loan_type)
            continue
        if ui_value is None:
            processed_inputs_dict[key] = default_value
            continue

        try:
            if key in percentage_decimal_keys:
                processed_inputs_dict[key] = convert_to_internal(ui_value, True)
            elif key in percentage_direct_keys:
                processed_inputs_dict[key] = convert_to_internal(ui_value, False)
            elif default_type == int:
                processed_inputs_dict[key] = int(round(float(ui_value)))
            elif default_type == float:
                processed_inputs_dict[key] = float(ui_value)
            elif default_type == bool:
                if isinstance(ui_value, str):
                    processed_inputs_dict[key] = ui_value.lower() == 'true'
                else:
                    processed_inputs_dict[key] = bool(ui_value)
            else:
                processed_inputs_dict[key] = ui_value
        except Exception as e:
            logger.error(f"Error processing input key '{key}': {e}")
            processed_inputs_dict[key] = default_value
            inputs_valid_for_processing = False
    processed_inputs_dict["refi_new_ltv"] = st.session_state.get(
        "input_refi_new_ltv",
        default_inputs_instance.refi_new_ltv * 100.0
    ) / 100.0
    processed_inputs_dict["refi_costs_pct_loan"] = st.session_state.get(
        "input_refi_costs_pct_loan",
        default_inputs_instance.refi_costs_pct_loan * 100.0
    ) / 100.0

    # --- Update session state only if necessary ---
    current_session_inputs_dict = st.session_state.get("inputs", {})
    if not all(np.isclose(processed_inputs_dict.get(k, np.nan), current_session_inputs_dict.get(k, np.nan), rtol=FLOAT_RTOL, atol=FLOAT_ATOL, equal_nan=True) if isinstance(processed_inputs_dict.get(k), float) else processed_inputs_dict.get(k) == current_session_inputs_dict.get(k) for k in all_input_keys):
        logger.info("Processed inputs differ from session state. Updating st.session_state['inputs'].")
        st.session_state["inputs"] = processed_inputs_dict.copy()

    # --- Create SimulationInputs Object ---
    sim_inputs_obj: Optional[SimulationInputs] = None
    inputs_valid_for_snapshot = False
    try:
        sim_inputs_obj = SimulationInputs(**st.session_state["inputs"])
        inputs_valid_for_snapshot = True
        logger.info("Successfully created SimulationInputs object from session state.")
    except Exception as e:
        st.error(f"Input Configuration Error: {e}. Cannot proceed. Please check sidebar inputs.")
        logger.error(f"Failed to create SimulationInputs object from session state: {e}", exc_info=True)

    # --- Simulation Execution ---
    processed_results = st.session_state.get("processed_results")

    if run_sim_button:
        logger.info("Run Simulation button clicked.")
        st.session_state["processed_results"] = None
        processed_results = None

        if sim_inputs_obj and inputs_valid_for_snapshot:
            sim_inputs_to_run = sim_inputs_obj
            num_sims_run = sim_inputs_to_run.num_simulations
            inputs_valid_for_sim = True
            logger.info("Using pre-validated SimulationInputs object for simulation run.")
        else:
            if not inputs_valid_for_snapshot:
                st.error("Cannot run simulation due to input configuration errors identified earlier.")
            else:
                st.error("Cannot run simulation: Input object not available.")
            logger.error("Simulation run aborted because SimulationInputs object creation failed or object unavailable.")
            inputs_valid_for_sim = False

        if not data_loaded_ok:
            st.error("Cannot run simulation: Forward curve data failed to load.")
            inputs_valid_for_sim = False

        if inputs_valid_for_sim and sim_inputs_to_run.enable_refinancing:
            if sim_inputs_to_run.refi_year > sim_inputs_to_run.hold_period:
                st.error(f"Input Error: Refinance Year ({sim_inputs_to_run.refi_year}) cannot be greater than Hold Period ({sim_inputs_to_run.hold_period}).")
                inputs_valid_for_sim = False

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
                logger.error(f"Sim execution failed: {error_message}")
                st.session_state["processed_results"] = None
            else:
                st.session_state["processed_results"] = mc_results
                logger.info("Simulation finished successfully.")
            st.rerun()
        else:
            st.warning("Simulation not run due to input errors or missing data.")

    # --- Display Results ---
    st.markdown("---")

    if processed_results and "error" in processed_results:
        st.error(f"Simulation Error: {processed_results['error']}")
    elif processed_results is None:
        if inputs_valid_for_snapshot:
            st.info("Adjust inputs in the sidebar and click 'Run Simulation' to see results.")
    elif processed_results:
        try:
            inputs_used_for_run = SimulationInputs(**st.session_state['inputs'])
            inputs_valid_for_display = True
        except Exception as e:
            st.error(f"Error recreating inputs object for display: {e}")
            inputs_valid_for_display = False
            inputs_used_for_run = None

        # Unpack results
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

        # Define tabs
        tab_keys = ["📊 Summary", "📈 IRR", "✖️ Multiples", "💰 Pro-Forma", "📉 Dynamics", "🛡️ Risk", "🔍 Audit", "🚪 Exit", "🔎 Sensitivity", "🗂️ Scenarios", "ℹ️ Guide"]
        tabs = st.tabs(tab_keys)

        # --- Populate Tabs ---
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

            st.markdown("---")
            col_mult_lev, col_mult_unlev = st.columns(2)

            def format_multiple(value):
                return f"{value:.1f}x" if value is not None and np.isfinite(value) else "N/A"

            with col_mult_lev:
                st.markdown("###### Levered Equity Multiple")
                mean_l_mult = metrics.get("mean_levered_multiple", np.nan)
                median_l_mult = metrics.get("median_levered_multiple", np.nan)
                p05_l_mult = metrics.get("p05_levered_multiple", np.nan)
                st.metric("Mean", format_multiple(mean_l_mult), help="Average [ Sum(+ Levered CFs) / (Initial Equity + Sum(|- Levered CFs|) ) ]")
                st.metric("Median", format_multiple(median_l_mult))
                st.metric("5th Pctl", format_multiple(p05_l_mult))

            with col_mult_unlev:
                st.markdown("###### Unlevered Equity Multiple")
                mean_u_mult = metrics.get("mean_unlevered_multiple", np.nan)
                median_u_mult = metrics.get("median_unlevered_multiple", np.nan)
                p05_u_mult = metrics.get("p05_unlevered_multiple", np.nan)
                st.metric("Mean", format_multiple(mean_u_mult), help="Average [ Sum(+ Unlevered CFs) / (Purchase Price + Sum(|- Unlevered CFs|) ) ]")
                st.metric("Median", format_multiple(median_u_mult))
                st.metric("5th Pctl", format_multiple(p05_u_mult))

            st.markdown("---")
            st.subheader("Year 0 / Initial State Snapshot (Based on Run Inputs)")
            if inputs_used_for_run and inputs_valid_for_display:
                try:
                    purchase_price = inputs_used_for_run.purchase_price
                    loan_to_cost = inputs_used_for_run.loan_to_cost
                    initial_loan_proceeds = purchase_price * loan_to_cost
                    initial_loan_costs = initial_loan_proceeds * inputs_used_for_run.initial_loan_costs_pct
                    net_initial_loan_proceeds = initial_loan_proceeds - initial_loan_costs
                    initial_equity = purchase_price - net_initial_loan_proceeds

                    potential_gross_rent = inputs_used_for_run.num_units * inputs_used_for_run.base_rent * 12
                    vacancy_amount = potential_gross_rent * inputs_used_for_run.current_vacancy
                    effective_gross_rent = potential_gross_rent - vacancy_amount
                    other_income_val = inputs_used_for_run.mean_other_income
                    effective_gross_income = effective_gross_rent + other_income_val
                    op_ex = inputs_used_for_run.mean_expense
                    net_operating_income = inputs_used_for_run.initial_noi
                    initial_cap_rate = net_operating_income / purchase_price if purchase_price > FLOAT_ATOL else np.nan
                    capex_yr1 = inputs_used_for_run.initial_capex

                    initial_monthly_payment_snapshot = 0.0
                    if not inputs_used_for_run.is_variable_rate and inputs_used_for_run.loan_type == LOAN_TYPE_AMORT:
                        rate_m = inputs_used_for_run.interest_rate / MONTHS_PER_YEAR
                        periods = inputs_used_for_run.loan_term_yrs * MONTHS_PER_YEAR
                        loan_amt = inputs_used_for_run.loan_amount
                        if periods > 0 and loan_amt > FLOAT_ATOL:
                            if abs(rate_m) > FLOAT_ATOL:
                                try:
                                    initial_monthly_payment_snapshot = npf.pmt(rate_m, periods, -loan_amt)
                                except Exception:
                                    initial_monthly_payment_snapshot = 0.0
                            else:
                                initial_monthly_payment_snapshot = loan_amt / periods
                        if not np.isfinite(initial_monthly_payment_snapshot):
                            initial_monthly_payment_snapshot = 0.0

                    interest_yr1, principal_yr1, _, effective_rate_yr1, _ = calculate_debt_service(
                        current_loan_type=inputs_used_for_run.loan_type,
                        current_interest_rate=inputs_used_for_run.interest_rate,
                        current_is_variable_rate=inputs_used_for_run.is_variable_rate,
                        current_balance=inputs_used_for_run.loan_amount,
                        monthly_payment=initial_monthly_payment_snapshot,
                        year=1,
                        sofr_spread=inputs_used_for_run.sofr_spread,
                        forward_curve=forward_curve_data,
                        std_dev_curve=std_dev_curve_data,
                        sofr_floor=inputs_used_for_run.sofr_floor,
                        rate_persistence_phi=inputs_used_for_run.rate_persistence_phi,
                        volatility_scalar=inputs_used_for_run.volatility_scalar,
                        prev_volatile_sofr_comp=None
                    )
                    interest_yr1 = interest_yr1 if np.isfinite(interest_yr1) else 0.0
                    principal_yr1 = principal_yr1 if np.isfinite(principal_yr1) else 0.0
                    total_debt_service_yr1 = interest_yr1 + principal_yr1

                    cfbds_yr1_approx = net_operating_income - capex_yr1
                    levered_cf_yr1_approx = cfbds_yr1_approx - total_debt_service_yr1
                    levered_cash_yield = levered_cf_yr1_approx / initial_equity if initial_equity > FLOAT_ATOL else np.nan

                    col_m1, col_m2, col_m3 = st.columns(3)
                    col_m1.metric("Purchase Price", f"${purchase_price:,.0f}")
                    col_m2.metric(f"Loan Amount ({loan_to_cost:.0%})", f"${initial_loan_proceeds:,.0f}")
                    col_m3.metric("Est. Initial Equity", f"${initial_equity:,.0f}")

                    col_w1, col_w2 = st.columns(2)

                    with col_w1:
                        fig_equity = go.Figure(go.Waterfall(
                            name="Equity", orientation="v", measure=["absolute", "relative", "total"],
                            x=["Purchase Price", "Net Loan Proceeds", "Initial Equity"],
                            y=[purchase_price, -net_initial_loan_proceeds, initial_equity],
                            connector={"line": {"color": "rgb(63, 63, 63)"}},
                            increasing={"marker": {"color": "rgba(50, 171, 96, 0.7)"}},
                            decreasing={"marker": {"color": "rgba(255, 127, 14, 0.7)"}},
                            totals={"marker": {"color": "rgba(55, 128, 191, 0.7)"}}
                        ))
                        fig_equity.update_layout(
                            title=dict(text='Initial Equity Breakdown', x=0.5, xanchor='center', font=dict(size=14)),
                            waterfallgap=0.2,
                            height=320,
                            margin=dict(t=40, b=10, l=10, r=10)
                        )
                        fig_equity.update_yaxes(tickprefix="$", tickformat=",.0f")
                        st.plotly_chart(fig_equity, use_container_width=True)

                    with col_w2:
                        fig_cf = go.Figure(go.Waterfall(
                            name="CF Yr1", orientation="v",
                            measure=["absolute", "relative", "relative", "relative", "total"],
                            x=["EGI", "OpEx", "CapEx", "Debt Service", "Levered CF"],
                            y=[effective_gross_income, -op_ex, -capex_yr1, -total_debt_service_yr1, levered_cf_yr1_approx],
                            connector={"line": {"color": "rgb(63, 63, 63)"}},
                            increasing={"marker": {"color": "rgba(50, 171, 96, 0.7)"}},
                            decreasing={"marker": {"color": "rgba(255, 127, 14, 0.7)"}},
                            totals={"marker": {"color": "rgba(55, 128, 191, 0.7)"}}
                        ))
                        fig_cf.update_layout(
                            title=dict(text='Est. Year 1 Levered Cash Flow', x=0.5, xanchor='center', font=dict(size=14)),
                            waterfallgap=0.2,
                            height=320,
                            margin=dict(t=40, b=10, l=10, r=10)
                        )
                        fig_cf.update_yaxes(tickprefix="$", tickformat=",.0f")
                        st.plotly_chart(fig_cf, use_container_width=True)

                    st.markdown("---")
                    col_m4, col_m5, col_m6 = st.columns(3)
                    col_m4.metric("Est. Net Operating Income (NOI)", f"${net_operating_income:,.0f}")
                    col_m5.metric("Initial Going-In Cap Rate", f"{initial_cap_rate:.2%}" if np.isfinite(initial_cap_rate) else "N/A")
                    col_m6.metric("Est. Levered Cash Yield (Yr 1)", f"{levered_cash_yield:.1%}" if np.isfinite(levered_cash_yield) else "N/A")
                except Exception as e:
                    st.error(f"Error calculating Initial State Snapshot: {e}")
                    logger.error(f"Initial State Calc Error after run: {e}", exc_info=True)
            else:
                st.warning("Cannot display Initial State Snapshot as input object could not be recreated.")

        with tabs[tab_keys.index("📈 IRR")]:
            st.subheader("IRR Distribution Analysis")
            mean_unlevered_irr = metrics.get("mean_unlevered_irr", np.nan)
            median_unlevered_irr = metrics.get("median_unlevered_irr", np.nan)
            p05_unlevered_irr = metrics.get("p05_unlevered_irr", np.nan)
            p95_unlevered_irr = metrics.get("p95_unlevered_irr", np.nan)
            mean_l_irr = metrics.get("mean_levered_irr", np.nan)
            median_l_irr = metrics.get("median_levered_irr", np.nan)
            p05_l_irr = metrics.get("p05_levered_irr", np.nan)
            p95_l_irr = metrics.get("p95_levered_irr", np.nan)
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Mean Unlevered IRR", f"{mean_unlevered_irr:.1%}" if np.isfinite(mean_unlevered_irr) else "N/A")
            col2.metric("Median Unlevered IRR", f"{median_unlevered_irr:.1%}" if np.isfinite(median_unlevered_irr) else "N/A")
            col3.metric("5th Pctl Unlevered IRR", f"{p05_unlevered_irr:.1%}" if np.isfinite(p05_unlevered_irr) else "N/A")
            col4.metric("95th Pctl Unlevered IRR", f"{p95_unlevered_irr:.1%}" if np.isfinite(p95_unlevered_irr) else "N/A")
            col5, col6, col7, col8 = st.columns(4)
            col5.metric("Mean Levered IRR", f"{mean_l_irr:.1%}" if np.isfinite(mean_l_irr) else "N/A")
            col6.metric("Median Levered IRR", f"{median_l_irr:.1%}" if np.isfinite(median_l_irr) else "N/A")
            col7.metric("5th Pctl Levered IRR", f"{p05_l_irr:.1%}" if np.isfinite(p05_l_irr) else "N/A")
            col8.metric("95th Pctl Levered IRR", f"{p95_l_irr:.1%}" if np.isfinite(p95_l_irr) else "N/A")
            st.markdown("---")
            st.subheader("IRR Distributions")
            unlev_failures = processed_results.get("unlev_forced_failures", 0)

            unlev_title = f"Distribution of Unlevered IRRs ({len(finite_unlevered)} valid runs)"
            if unlev_failures > 0:
                unlev_title += f" — {unlev_failures} calculation failures"
            if finite_unlevered:
                fig_unlevered = plot_irr_distribution(
                    irr_values=finite_unlevered,
                    title=unlev_title,
                    mean_irr=mean_unlevered_irr,
                    median_irr=median_unlevered_irr,
                    p05_irr=p05_unlevered_irr,
                    p95_irr=p95_unlevered_irr,
                    color='cornflowerblue',
                    percent_format=True
                )
                st.plotly_chart(fig_unlevered, use_container_width=True)
                st.caption("This chart shows the distribution of potential returns based purely on property operations and sale, before considering debt. The spread reflects uncertainty from market factors (rent, vacancy, exit cap) and operating assumptions.")
            else:
                st.warning('Unlevered IRR data not available.')

            st.markdown("---")
            lev_failures = processed_results.get("lev_forced_failures", 0)

            lev_title = f"Distribution of Levered IRRs ({len(finite_levered)} valid runs)"
            if lev_failures > 0:
                lev_title += f" — {lev_failures} calculation failures"
            if finite_levered:
                st.markdown("#### Levered IRR Distribution (Full View)")
                fig_levered_full = plot_irr_distribution(
                    irr_values=finite_levered,
                    title=lev_title,
                    mean_irr=mean_l_irr,
                    median_irr=median_l_irr,
                    p05_irr=p05_l_irr,
                    p95_irr=p95_l_irr,
                    color='darkorange',
                    percent_format=True
                )
                st.plotly_chart(fig_levered_full, use_container_width=True)
                st.caption("This distribution includes the impact of financing (initial loan, debt service, refinancing, payoff). Leverage typically widens the distribution, increasing potential upside and downside compared to the Unlevered IRR. Compare the Mean and Median to gauge skewness.")

                st.markdown("---")
                st.markdown("#### Zoomed-In View of Levered IRRs")
                x_range_zoom = None
                zoom_title = "Zoomed View: Levered IRRs (Error Calculating Range)"
                try:
                    if len(finite_levered) >= 2:
                        low_bound = np.percentile(finite_levered, 1)
                        high_bound = np.percentile(finite_levered, 99)
                        spread = high_bound - low_bound
                        zoom_min = max(-0.50, low_bound - spread * 0.1)
                        zoom_max = min(1.00, high_bound + spread * 0.1)
                        if zoom_max > zoom_min + 0.01:
                            x_range_zoom = (zoom_min, zoom_max)
                            zoom_title = f"Zoomed View: Levered IRRs ({x_range_zoom[0]:.0%} to {x_range_zoom[1]:.0%})"
                        else:
                            st.caption("Zoomed IRR plot skipped: Range too narrow after calculating percentiles.")
                            x_range_zoom = None
                    else:
                        st.caption("Zoomed IRR plot skipped: Not enough data points.")
                        x_range_zoom = None

                    if x_range_zoom:
                        fig_levered_zoom = plot_irr_distribution(
                            irr_values=finite_levered,
                            title=zoom_title,
                            mean_irr=mean_l_irr,
                            median_irr=median_l_irr,
                            p05_irr=p05_l_irr,
                            p95_irr=metrics.get('p95_levered_irr', np.nan),
                            color='coral',
                            percent_format=True,
                            x_range=x_range_zoom,
                            bins=30
                        )
                        st.plotly_chart(fig_levered_zoom, use_container_width=True)
                        st.caption("This zoomed view focuses on the central part of the Levered IRR distribution, excluding extreme outliers, to better visualize the most likely range of outcomes.")
                except Exception as e:
                    logger.error(f"Error creating zoomed IRR plot: {e}")
                    st.caption(f"Could not generate zoomed IRR plot: {e}")
            else:
                st.warning('Levered IRR data not available.')

        with tabs[tab_keys.index("✖️ Multiples")]:
            st.subheader("Equity Multiple Distribution Analysis")
            st.markdown(
                "The Equity Multiple measures total **positive** cash returned over the hold period divided by the total capital invested "
                "(defined here as the initial investment plus the sum of all subsequent **negative** cash flows). "
                "Unlike IRR, it is not time-sensitive but shows the magnitude of profit relative to the total capital required."
            )
            st.markdown("---")

            finite_unlevered_multiples = processed_results.get("finite_unlevered_multiples", [])
            finite_levered_multiples = processed_results.get("finite_levered_multiples", [])

            st.markdown("#### Unlevered Multiple Distribution")
            mean_u_mult = metrics.get("mean_unlevered_multiple", np.nan)
            median_u_mult = metrics.get("median_unlevered_multiple", np.nan)
            p05_u_mult = metrics.get("p05_unlevered_multiple", np.nan)
            p95_u_mult = metrics.get("p95_unlevered_multiple", np.nan)
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Mean", f"{mean_u_mult:.1f}x" if np.isfinite(mean_u_mult) else "N/A")
            col2.metric("Median", f"{median_u_mult:.1f}x" if np.isfinite(median_u_mult) else "N/A")
            col3.metric("5th Pctl", f"{p05_u_mult:.1f}x" if np.isfinite(p05_u_mult) else "N/A")
            col4.metric("95th Pctl", f"{p95_u_mult:.1f}x" if np.isfinite(p95_u_mult) else "N/A")
            unlev_mult_title = f"Distribution of Unlevered Multiples ({len(finite_unlevered_multiples)} valid runs)"

            if finite_unlevered_multiples:
                try:
                    fig_unlev_mult = plot_multiple_distribution(
                        multiple_values=finite_unlevered_multiples,
                        title=unlev_mult_title,
                        mean_multiple=mean_u_mult,
                        median_multiple=median_u_mult,
                        p05_multiple=p05_u_mult,
                        p95_multiple=p95_u_mult,
                        color='skyblue'
                    )
                    st.plotly_chart(fig_unlev_mult, use_container_width=True)
                    st.caption(
                        "Shows the distribution of the Unlevered Equity Multiple (Sum Positive Unlevered CFs / (Purchase Price + Sum |Negative Unlevered CFs|)). "
                        "This reflects the total return on the property itself, independent of financing structure."
                    )
                except Exception as e_unl_mult_plot:
                    st.warning(f"Could not plot Unlevered Multiple distribution: {e_unl_mult_plot}")
                    logger.error(f"Unlevered Multiple plot error: {e_unl_mult_plot}", exc_info=True)
            else:
                st.warning('Unlevered Multiple data not available.')

            st.markdown("---")

            st.markdown("#### Levered Multiple Distribution")
            mean_l_mult = metrics.get("mean_levered_multiple", np.nan)
            median_l_mult = metrics.get("median_levered_multiple", np.nan)
            p05_l_mult = metrics.get("p05_levered_multiple", np.nan)
            p95_l_mult = metrics.get("p95_levered_multiple", np.nan)
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Mean", f"{mean_l_mult:.1f}x" if np.isfinite(mean_l_mult) else "N/A")
            col2.metric("Median", f"{median_l_mult:.1f}x" if np.isfinite(median_l_mult) else "N/A")
            col3.metric("5th Pctl", f"{p05_l_mult:.1f}x" if np.isfinite(p05_l_mult) else "N/A")
            col4.metric("95th Pctl", f"{p95_l_mult:.1f}x" if np.isfinite(p95_l_mult) else "N/A")
            lev_mult_title = f"Distribution of Levered Multiples ({len(finite_levered_multiples)} valid runs)"

            if finite_levered_multiples:
                try:
                    fig_lev_mult = plot_multiple_distribution(
                        multiple_values=finite_levered_multiples,
                        title=lev_mult_title,
                        mean_multiple=mean_l_mult,
                        median_multiple=median_l_mult,
                        p05_multiple=p05_l_mult,
                        p95_multiple=p95_l_mult,
                        color='lightcoral'
                    )
                    st.plotly_chart(fig_lev_mult, use_container_width=True)
                    st.caption(
                        "Shows the distribution of the Levered Equity Multiple (Sum Positive Levered CFs / (Initial Equity + Sum |Negative Levered CFs|)). "
                        "This reflects the total return to the equity investor after accounting for debt financing and refinancing effects."
                    )
                except Exception as e_lev_mult_plot:
                    st.warning(f"Could not plot Levered Multiple distribution: {e_lev_mult_plot}")
                    logger.error(f"Levered Multiple plot error: {e_lev_mult_plot}", exc_info=True)
            else:
                st.warning('Levered Multiple data not available.')

        with tabs[tab_keys.index("💰 Pro-Forma")]:
            st.subheader("Average Annual Pro-Forma Cash Flows")
            st.info(
                "This table displays the **average** cash flows across all completed simulation runs for each year of the hold period, "
                "plus Year 0 setup and the final sale year. It provides a central tendency view of the potential financial performance."
            )
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
                total_years = hold_period_actual + 1

                initial_proceeds_net_row = [net_initial_loan_proceeds] + [0.0] * hold_period_actual
                refi_proceeds_net_row = [0.0] * total_years
                refi_payoff_row = [0.0] * total_years
                final_payoff_row = [0.0] * total_years

                avg_refi_proceeds_net = 0.0
                avg_refi_loan_payoff = 0.0
                if refi_year:
                    avg_refi_proceeds_net_list = avg_cash_flow_data.get("refi_proceeds_net", [0.0] * hold_period_actual)
                    avg_loan_balance_list = avg_cash_flow_data.get("loan_balance", [0.0] * hold_period_actual)

                    if len(avg_refi_proceeds_net_list) >= refi_year:
                        avg_refi_proceeds_net = avg_refi_proceeds_net_list[refi_year - 1]
                        refi_proceeds_net_row[refi_year] = avg_refi_proceeds_net
                    else:
                        logger.warning(f"Pro-Forma: Refi year {refi_year} out of bounds for avg refi proceeds list.")

                    if refi_year > 1 and len(avg_loan_balance_list) >= refi_year - 1:
                        avg_refi_loan_payoff = avg_loan_balance_list[refi_year - 2]
                        refi_payoff_row[refi_year] = -avg_refi_loan_payoff
                    elif refi_year == 1:
                        avg_refi_loan_payoff = initial_loan_proceeds
                        refi_payoff_row[refi_year] = -avg_refi_loan_payoff
                    else:
                        logger.warning(f"Pro-Forma: Cannot determine avg refi loan payoff for year {refi_year}.")

                net_sale_proceeds = metrics.get("mean_exit_value", 0.0)
                final_loan_payoff = avg_cash_flow_data.get("loan_balance", [0.0])[-1] if avg_cash_flow_data.get("loan_balance") else 0.0
                final_payoff_row[-1] = -final_loan_payoff

                last_avg_unlev_cf = avg_cash_flow_data.get("unlevered_cf", [0.0])[-1] if avg_cash_flow_data.get("unlevered_cf") else 0.0
                last_avg_lev_cf = avg_cash_flow_data.get("levered_cf", [0.0])[-1] if avg_cash_flow_data.get("levered_cf") else 0.0
                final_avg_unlev_cf_for_irr = last_avg_unlev_cf + net_sale_proceeds
                final_avg_lev_cf_for_irr = last_avg_lev_cf + net_sale_proceeds - final_loan_payoff

                avg_noi = avg_cash_flow_data.get("noi", [np.nan]*hold_period_actual)
                avg_capex = avg_cash_flow_data.get("capex", [np.nan]*hold_period_actual)
                avg_interest = avg_cash_flow_data.get("interest", [np.nan]*hold_period_actual)
                avg_principal = avg_cash_flow_data.get("principal", [np.nan]*hold_period_actual)

                cfbds_row_vals = [(n - c) if pd.notna(n) and pd.notna(c) else np.nan for n, c in zip(avg_noi, avg_capex)]
                total_debt_service_vals = [-(abs(i) + abs(p)) if pd.notna(i) and pd.notna(p) else np.nan for i, p in zip(avg_interest, avg_principal)]

                proforma_rows = [
                    ("--- Unlevered Cash Flows ---", [np.nan] * total_years),
                    ("Purchase Price", [-purchase_price] + [np.nan] * hold_period_actual),
                    ("Potential Gross Rent (PGR)", [np.nan] + avg_cash_flow_data.get("potential_rent", [np.nan]*hold_period_actual)),
                    ("Less: Vacancy Loss", [np.nan] + [-abs(v) if pd.notna(v) else np.nan for v in avg_cash_flow_data.get("vacancy_loss", [np.nan]*hold_period_actual)]),
                    ("Effective Gross Rent (EGR)", [np.nan] + avg_cash_flow_data.get("egr", [np.nan]*hold_period_actual)),
                    ("Plus: Other Income", [np.nan] + avg_cash_flow_data.get("other_income", [np.nan]*hold_period_actual)),
                    ("Effective Gross Income (EGI)", [np.nan] + avg_cash_flow_data.get("egi", [np.nan]*hold_period_actual)),
                    ("Less: Operating Expenses", [np.nan] + [-abs(e) if pd.notna(e) else np.nan for e in avg_cash_flow_data.get("expenses", [np.nan]*hold_period_actual)]),
                    ("Net Operating Income (NOI)", [np.nan] + avg_noi),
                    ("Less: Capital Expenditures (CapEx)", [np.nan] + [-abs(c) if pd.notna(c) else np.nan for c in avg_capex]),
                    ("Cash Flow Before Debt Service (CFBDS)", [np.nan] + cfbds_row_vals),
                    ("--- Levered Cash Flows & Debt ---", [np.nan] * total_years),
                    ("Net Loan Proceeds", initial_proceeds_net_row),
                    ("Net Refinancing Proceeds", refi_proceeds_net_row),
                    ("Loan Payoff", final_payoff_row),
                    ("Interest Paid", [np.nan] + [-abs(i) if pd.notna(i) else np.nan for i in avg_interest]),
                    ("Principal Paid", [np.nan] + [-abs(p) if pd.notna(p) else np.nan for p in avg_principal]),
                    ("Total Debt Service", [np.nan] + total_debt_service_vals),
                    ("--- Sale ---", [np.nan] * total_years),
                    ("Net Sale Proceeds", [np.nan] * hold_period_actual + [net_sale_proceeds]),
                    ("--- Cash Flows for IRR ---", [np.nan] * total_years),
                    ("Unlevered Cash Flow (IRR)", [-purchase_price] + avg_cash_flow_data.get("unlevered_cf", [0.0] * hold_period_actual)[:-1] + [final_avg_unlev_cf_for_irr]),
                    ("Levered Cash Flow (IRR)", [-initial_equity] + avg_cash_flow_data.get("levered_cf", [0.0] * hold_period_actual)[:-1] + [final_avg_lev_cf_for_irr]),
                    ("--- Other Info ---", [np.nan] * total_years),
                    ("End of Year Loan Balance", [initial_loan_proceeds] + avg_cash_flow_data.get("loan_balance", [np.nan]*hold_period_actual))
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
                        if abs(num_val) < 1e-3:
                            return "$0"
                        fmt_str = "${:,.0f}" if abs(num_val) >= 1 else "${:,.2f}"
                        return fmt_str.format(num_val) if num_val >= 0 else f"({fmt_str.format(abs(num_val))})"
                    except (TypeError, ValueError):
                        return str(val)
                st.dataframe(
                    proforma_df.style.format(fmt_proforma, na_rep="-")
                    .set_properties(**{"text-align": "right"})
                    .set_table_styles([{"selector": "th", "props": [("text-align", "left")]}]),
                    use_container_width=True
                )

                st.markdown("##### Returns Calculated from Average Cash Flow Stream")
                avg_unlev_irr_display = "N/A"
                avg_lev_irr_display = "N/A"
                avg_unlev_mult_display = "N/A"
                avg_lev_mult_display = "N/A"

                def format_multiple_local(value):
                    return f"{value:.1f}x" if value is not None and np.isfinite(value) else "N/A"

                try:
                    unlev_irr_label = "Unlevered Cash Flow (IRR)"
                    lev_irr_label = "Levered Cash Flow (IRR)"
                    avg_unlev_stream_vals = proforma_df.loc[unlev_irr_label].astype(float).tolist()
                    avg_lev_stream_vals = proforma_df.loc[lev_irr_label].astype(float).tolist()

                    if len(avg_unlev_stream_vals) >= 2 and all(np.isfinite(avg_unlev_stream_vals)) and any(x < 0 for x in avg_unlev_stream_vals) and any(x > 0 for x in avg_unlev_stream_vals):
                        avg_unlev_irr = npf.irr(avg_unlev_stream_vals)
                        avg_unlev_irr_display = f"{avg_unlev_irr:.1%}" if np.isfinite(avg_unlev_irr) else "Calc Error"
                    else:
                        logger.warning("ProForma Unlevered cash flow stream invalid for IRR.")

                    if len(avg_lev_stream_vals) >= 2 and all(np.isfinite(avg_lev_stream_vals)) and any(x < 0 for x in avg_lev_stream_vals) and any(x > 0 for x in avg_lev_stream_vals):
                        avg_lev_irr = npf.irr(avg_lev_stream_vals)
                        avg_lev_irr_display = f"{avg_lev_irr:.1%}" if np.isfinite(avg_lev_irr) else "Calc Error"
                    else:
                        logger.warning("ProForma Levered cash flow stream invalid for IRR.")

                    avg_initial_investment = abs(avg_unlev_stream_vals[0]) if len(avg_unlev_stream_vals) > 0 else 0.0
                    avg_total_positive_unlevered = sum(cf for cf in avg_unlev_stream_vals[1:] if cf > 0 and np.isfinite(cf))
                    avg_total_negative_unlevered_abs = sum(abs(cf) for cf in avg_unlev_stream_vals[1:] if cf < 0 and np.isfinite(cf))
                    avg_unlevered_denominator = avg_initial_investment + avg_total_negative_unlevered_abs
                    if avg_unlevered_denominator > FLOAT_ATOL:
                        avg_unlev_mult = avg_total_positive_unlevered / avg_unlevered_denominator
                        avg_unlev_mult_display = format_multiple_local(avg_unlev_mult)
                    else:
                        logger.warning("ProForma Unlev Mult calc failed: Denominator near zero.")
                        avg_unlev_mult_display = "N/A"

                    avg_initial_equity = abs(avg_lev_stream_vals[0]) if len(avg_lev_stream_vals) > 0 else 0.0
                    avg_total_positive_levered = sum(cf for cf in avg_lev_stream_vals[1:] if cf > 0 and np.isfinite(cf))
                    avg_total_negative_levered_abs = sum(abs(cf) for cf in avg_lev_stream_vals[1:] if cf < 0 and np.isfinite(cf))
                    avg_levered_denominator = avg_initial_equity + avg_total_negative_levered_abs
                    if avg_levered_denominator > FLOAT_ATOL:
                        avg_lev_mult = avg_total_positive_levered / avg_levered_denominator
                        avg_lev_mult_display = format_multiple_local(avg_lev_mult)
                    else:
                        logger.warning("ProForma Lev Mult calc failed: Denominator near zero.")
                        avg_lev_mult_display = "N/A"

                except KeyError as ke:
                    logger.error(f"IRR/Mult calculation failed: Row label not found - {ke}. Check proforma_rows definitions.")
                    st.warning(f"Could not calculate average IRRs/Multiples due to missing row: {ke}")
                except Exception as e:
                    logger.error(f"Error calculating average IRR/Mult: {e}")
                    avg_unlev_irr_display = "Error"
                    avg_lev_irr_display = "Error"
                    avg_unlev_mult_display = "Error"
                    avg_lev_mult_display = "Error"

                col1, col2 = st.columns(2)
                col1.metric("Unlevered IRR (Avg CF)", avg_unlev_irr_display)
                col2.metric("Levered IRR (Avg CF)", avg_lev_irr_display)

                col3, col4 = st.columns(2)
                col3.metric("Unlevered Multiple (Avg CF)", avg_unlev_mult_display, help="Sum(+ Unlev CFs) / (Init Inv + Sum(|- Unlev CFs|)) from Avg Stream")
                col4.metric("Levered Multiple (Avg CF)", avg_lev_mult_display, help="Sum(+ Lev CFs) / (Init Equity + Sum(|- Lev CFs|)) from Avg Stream")

                st.markdown("---")
                st.subheader("Key Assumptions & Metrics (Average by Year)")
                exit_cap_rate = metrics.get("mean_exit_cap", np.nan)
                assumption_rows_data = {
                    "Avg Market Rent ($/Unit/Mo)": avg_cash_flow_data.get("rent_per_unit", []),
                    "Vacancy Rate (%)": avg_cash_flow_data.get("vacancy_rate", []),
                    "Rent Growth (%)": avg_cash_flow_data.get("rent_growth_pct", []),
                    "Expense Growth (%)": avg_cash_flow_data.get("expense_growth_pct", []),
                    "CapEx Growth (%)": avg_cash_flow_data.get("capex_growth_pct", []),
                    ("Interest Rate (%)" if not inputs_used_for_run.is_variable_rate else "Effective Interest Rate (%)"): avg_cash_flow_data.get("interest_rates", [])
                }
                assumption_rows = [(label, data) for label, data in assumption_rows_data.items()]
                if np.isfinite(exit_cap_rate):
                    cap_rate_row = [np.nan] * (hold_period_actual - 1) + [exit_cap_rate]
                    assumption_rows.append(("Exit Cap Rate (%)", cap_rate_row))
                clean_assumptions = [(label, vals) for label, vals in assumption_rows if isinstance(vals, list) and len(vals) == hold_period_actual]
                if clean_assumptions:
                    assumption_df = pd.DataFrame({label: vals for label, vals in clean_assumptions}).T
                    assumption_df.columns = [f"Yr {i+1}" for i in range(hold_period_actual)]
                    assumption_df.index.name = "Assumption"
                    def format_assumption_value(val, label):
                        if pd.isna(val):
                            return "-"
                        try:
                            f_val = float(val)
                            if "Rent ($/Unit/Mo)" in label:
                                return f"${f_val:,.0f}"
                            elif ("Rate (%)" in label and "Growth" not in label):
                                return f"{f_val:.1%}"
                            elif "Growth (%)" in label:
                                return f"{f_val:.1f}%"
                            else:
                                return f"{f_val:.2f}"
                        except (ValueError, TypeError):
                            return str(val)
                    formatted_df = assumption_df.apply(lambda row: pd.Series([format_assumption_value(val, row.name) for val in row], index=row.index), axis=1)
                    st.dataframe(formatted_df.style.set_properties(**{"text-align": "right"}).set_table_styles([{"selector": "th", "props": [("text-align", "left")]}]), use_container_width=True)
                else:
                    st.warning("Could not calculate key assumptions by year.")

        with tabs[tab_keys.index("📉 Dynamics")]:
            st.subheader("Key Simulation Dynamics Over Time")
            years_list_plot = plot_data.get("years", [])
            rent_norm_plot_data = plot_data.get("rent_norm_plot", {})
            vacancy_plot_df = plot_data.get("vacancy_plot_df")
            scatter_data = plot_data.get("scatter_plot", {})
            inputs_for_run = st.session_state.get("inputs", {})

            st.markdown("#### Projected Market Rent vs. Fair Value Rent")
            if (rent_norm_plot_data and years_list_plot and
                all(k in rent_norm_plot_data for k in ["market_p05", "market_p50", "market_p95", "normal_p50"]) and
                all(len(rent_norm_plot_data[k]) == len(years_list_plot) for k in rent_norm_plot_data)):
                try:
                    fig_rent_norm = plot_rent_vs_normal(
                        years=years_list_plot,
                        market_p05=rent_norm_plot_data["market_p05"],
                        market_p50=rent_norm_plot_data["market_p50"],
                        market_p95=rent_norm_plot_data["market_p95"],
                        normal_p50=rent_norm_plot_data["normal_p50"]
                    )
                    st.plotly_chart(fig_rent_norm, use_container_width=True)
                    st.caption(
                        "This chart illustrates the simulation's rent dynamics. The blue band represents the 5th-95th percentile range "
                        "of simulated *Market Rent* actually used in cash flows, with the solid blue line as the median. "
                        "The purple dashed line shows the median simulated *Fair Value Rent* (the underlying trend). "
                        "Observe how the Market Rent converges towards the Fair Value Rent over the 'Years to Normalize Rent' input."
                    )
                except Exception as e:
                    st.warning(f"Could not plot Projected Market Rent vs Fair Value Rent: {e}")
                    logging.error(f"Rent vs Fair Value plot error: {e}", exc_info=True)
            else:
                st.warning("Rent path data is missing or invalid. Cannot display plot.")
            st.markdown("---")

            st.markdown("#### Vacancy Rate Distribution Over Time")
            if vacancy_plot_df is not None and not vacancy_plot_df.empty:
                try:
                    fig_vacancy_dist = plot_vacancy_distribution(vacancy_plot_df)
                    st.plotly_chart(fig_vacancy_dist, use_container_width=True)
                    st.caption(
                        "This box plot shows the simulated distribution of vacancy rates for each year. "
                        "The box represents the middle 50% of outcomes (25th to 75th percentile), the line inside is the median, "
                        "and the 'whiskers' typically extend to 1.5x the box height. Dots are outliers. "
                        "Wider boxes or more outliers indicate greater uncertainty in vacancy loss for that year."
                    )
                except Exception as e:
                    st.warning(f'Error plotting vacancy distribution: {e}')
                    logging.error(f"Vacancy distribution plot error: {e}", exc_info=True)
            else:
                st.warning('Vacancy distribution data not available or format error.')
            st.markdown("---")

            st.markdown("#### Terminal Year Rent Growth vs. Exit Cap Rate")
            if scatter_data and scatter_data.get("term_rent_growth_pct") and scatter_data.get("exit_cap_rate_pct"):
                try:
                    fig_scatter = plot_terminal_growth_vs_exit_cap(scatter_data)
                    st.plotly_chart(fig_scatter, use_container_width=True)
                    st.caption(
                        "This scatter plot shows the relationship between the simulated rent growth in the final year "
                        "and the simulated exit cap rate across all runs. The 'Exit Cap Adj. Factor' input influences this relationship. "
                        "A negative correlation (typical) suggests higher terminal growth is associated with lower exit caps (higher valuations), and vice-versa. "
                        "The red line shows the OLS trend."
                    )
                except Exception as e:
                    st.warning(f"Could not plot Terminal Growth vs Exit Cap: {e}")
                    logging.error(f"Terminal Growth vs Exit Cap plot error: {e}", exc_info=True)
            else:
                st.warning("Scatter plot data is missing or invalid.")
            st.markdown("---")

            if inputs_for_run.get("is_variable_rate", False):
                st.markdown("#### Simulated Underlying SOFR Rate Distribution")
                sim_results = processed_results.get("raw_results_for_audit", [])
                fixed_spread = inputs_for_run.get("sofr_spread", 0.0)
                hold_period_plot = len(years_list_plot)
                underlying_sofr_paths = []
                if hold_period_plot > 0 and sim_results:
                    for sim_result in sim_results:
                        effective_rates = sim_result.get("interest_rates", [])
                        if (isinstance(effective_rates, list) and
                            len(effective_rates) == hold_period_plot and
                            all(isinstance(r, (int, float)) and np.isfinite(r) for r in effective_rates)):
                            underlying_path = [r - fixed_spread for r in effective_rates]
                            underlying_sofr_paths.append(underlying_path)

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
                        refi_year_plot = inputs_used_for_run.refi_year if inputs_used_for_run.enable_refinancing else None
                        fig_sofr = plot_simulated_sofr_distribution(
                            years_list=years_list_plot,
                            underlying_sofr_paths=underlying_sofr_paths,
                            forward_rates_input=forward_rates_plot,
                            refi_year=refi_year_plot
                        )
                        st.plotly_chart(fig_sofr, use_container_width=True)
                        st.caption(
                            "This chart displays the range (5th-95th percentile band and median line) of the simulated underlying SOFR component "
                            "(including volatility, persistence, and floor, but *before* adding the spread). "
                            "It's compared against the input Forward SOFR Curve (dotted line). Deviations highlight potential interest rate risk exposure. "
                            "The dashed vertical line indicates the refinance year, if applicable."
                        )
                    except Exception as e:
                        st.warning(f"Could not generate SOFR distribution plot: {e}")
                        logging.error(f"Error plotting SOFR distribution: {e}", exc_info=True)
                elif hold_period_plot > 0:
                    st.info("No valid simulated interest rate data found for floating rate simulations. Check if 'interest_rates' key exists and contains valid data in simulation results.")
                st.markdown("---")

            if not inputs_for_run.get("is_variable_rate", False):
                st.markdown("---")
            st.markdown("#### Loan Balance & LTV Over Time")
            loan_balance_paths = get_valid_paths(sim_results_completed_audit, "loan_balance", inputs_used_for_run.hold_period)
            ltv_paths = get_valid_paths(sim_results_completed_audit, "ltv_estimate", inputs_used_for_run.hold_period)

            if loan_balance_paths and years_list_plot:
                if not ltv_paths:
                    logger.warning("LTV path data missing for Loan Balance/LTV plot.")
                    st.warning("Loan-to-Value (LTV) data is missing or invalid. Cannot display combined plot.")
                else:
                    try:
                        refi_year_plot = inputs_used_for_run.refi_year if inputs_used_for_run.enable_refinancing else None
                        fig_loan = plot_loan_balance_distribution(
                            years=years_list_plot,
                            loan_balance_paths=loan_balance_paths,
                            ltv_paths=ltv_paths,
                            refi_year=refi_year_plot
                        )
                        st.plotly_chart(fig_loan, use_container_width=True)
                        st.caption(
                            "This chart shows the distribution (median and 5th-95th percentile bands) of the End-of-Year Loan Balance (left axis, $) "
                            "and the estimated Loan-to-Value (LTV) ratio (right axis, %). Note how refinancing, if enabled, "
                            "can cause a distinct change (indicated by the dashed line) in both the loan balance and LTV in the specified refinance year."
                        )
                    except Exception as e:
                        st.warning(f"Could not plot Loan Balance / LTV: {e}")
                        logging.error(f"Loan Balance plot error: {e}", exc_info=True)
            else:
                st.warning("Loan Balance data not available for plotting.")

        with tabs[tab_keys.index("🛡️ Risk")]:
            st.subheader("Risk Profile & Risk-Adjusted Return Metrics")
            st.markdown("Metrics based on the distribution of **Levered IRR** outcomes from valid simulation runs.")
            if not finite_levered:
                st.warning("Levered IRR results are not available for risk analysis.")
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
                st.markdown("---")
                st.subheader("Levered IRR Distribution Visualization")
                failure_threshold = -0.99
                finite_levered_plot_data = [irr for irr in finite_levered if irr > failure_threshold]
                num_excluded = len(finite_levered) - len(finite_levered_plot_data)
                if finite_levered_plot_data:
                    try:
                        fig_box = go.Figure()
                        fig_box.add_trace(go.Box(
                            y=finite_levered_plot_data,
                            name='Levered IRR Dist.',
                            marker_color='rgba(255, 127, 14, 0.7)',
                            boxpoints='outliers',
                            jitter=0.3,
                            pointpos=-1.8,
                            hoverinfo='y'
                        ))
                        mean_filtered = np.mean(finite_levered_plot_data)
                        if np.isfinite(mean_filtered):
                            fig_box.add_trace(go.Scatter(
                                x=['Levered IRR Dist.'],
                                y=[mean_filtered],
                                mode='markers',
                                marker=dict(color='red', size=8, symbol='cross-thin'),
                                name='Mean (Filtered)',
                                hoverinfo='skip'
                            ))
                        fig_box.update_layout(
                            title="Box Plot of Calculated Levered IRR Outcomes",
                            yaxis_title="Levered IRR",
                            xaxis_title="",
                            template="plotly_white",
                            yaxis_tickformat=".1%",
                            showlegend=False
                        )
                        fig_box.update_xaxes(showticklabels=False)
                        st.plotly_chart(fig_box, use_container_width=True)
                        st.caption(
                            f"This box plot summarizes the Levered IRR distribution. The box shows the interquartile range (IQR, 25th-75th percentile), "
                            f"the line inside is the median, whiskers typically extend to 1.5x IQR, and dots are outliers. The red cross indicates the mean."
                            f"{f' Note: {num_excluded} extreme outlier(s) below {failure_threshold:.0%} excluded for visual clarity.' if num_excluded > 0 else ''}"
                        )
                    except Exception as e:
                        st.warning(f"Could not generate Levered IRR Box Plot: {e}")
                else:
                    st.info(f"No Levered IRR data available to display after filtering {num_excluded} outlier(s).")

        with tabs[tab_keys.index("🔍 Audit")]:
            st.subheader("Detailed Audit of Individual Simulation")
            st.info(
                "This tab shows the detailed annual cash flows and metrics for a **single selected simulation run**. "
                "Use the number input below to choose which specific simulation path (out of the total runs completed) you want to inspect."
            )
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
                    st.warning(f"Simulation {selected_sim_index+1} data is incomplete or has zero hold period.")
                else:
                    purchase_price = inputs_used_for_run.purchase_price
                    loan_to_cost = inputs_used_for_run.loan_to_cost
                    initial_loan_proceeds = purchase_price * loan_to_cost
                    initial_loan_costs = initial_loan_proceeds * inputs_used_for_run.initial_loan_costs_pct
                    net_initial_loan_proceeds = initial_loan_proceeds - initial_loan_costs
                    initial_equity = purchase_price - net_initial_loan_proceeds

                    refi_year = inputs_used_for_run.refi_year if inputs_used_for_run.enable_refinancing else None
                    total_years = hold_period_audit + 1

                    initial_proceeds_net_row = [net_initial_loan_proceeds] + [0.0] * hold_period_audit
                    refi_proceeds_net_row = [0.0] * total_years
                    refi_payoff_row = [0.0] * total_years
                    final_payoff_row = [0.0] * total_years

                    sim_refi_proceeds_net = 0.0
                    sim_refi_loan_payoff = 0.0
                    if refi_year:
                        sim_refi_proceeds_net_list = audit_sim.get("refi_proceeds_net", [0.0] * hold_period_audit)
                        sim_loan_balance_list = audit_sim.get("loan_balance", [0.0] * hold_period_audit)

                        if len(sim_refi_proceeds_net_list) >= refi_year:
                            sim_refi_proceeds_net = sim_refi_proceeds_net_list[refi_year - 1]
                            refi_proceeds_net_row[refi_year] = sim_refi_proceeds_net
                        else:
                            logger.warning(f"Audit Sim {selected_sim_index+1}: Refi year {refi_year} out of bounds for proceeds list.")

                        if refi_year > 1 and len(sim_loan_balance_list) >= refi_year - 1:
                            sim_refi_loan_payoff = sim_loan_balance_list[refi_year - 2]
                            refi_payoff_row[refi_year] = -sim_refi_loan_payoff
                        elif refi_year == 1:
                            sim_refi_loan_payoff = initial_loan_proceeds
                            refi_payoff_row[refi_year] = -sim_refi_loan_payoff
                        else:
                            logger.warning(f"Audit Sim {selected_sim_index+1}: Cannot determine refi loan payoff for year {refi_year}.")

                    net_sale_proceeds = audit_sim.get("exit_value_net", 0.0)
                    final_loan_payoff = audit_sim.get("loan_balance", [0.0])[-1] if audit_sim.get("loan_balance") else 0.0
                    final_payoff_row[-1] = -final_loan_payoff

                    last_unlev_cf = audit_sim.get("unlevered_cf", [0.0])[-1] if audit_sim.get("unlevered_cf") else 0.0
                    last_lev_cf = audit_sim.get("levered_cf", [0.0])[-1] if audit_sim.get("levered_cf") else 0.0
                    final_unlev_cf_for_irr = last_unlev_cf + net_sale_proceeds
                    final_lev_cf_for_irr = last_lev_cf + net_sale_proceeds - final_loan_payoff

                    def _pad_list(data_list: Optional[List[Any]], expected_len: int, pad_value: Any = np.nan) -> List[Any]:
                        if data_list is None:
                            return [pad_value] * expected_len
                        base_list = list(data_list) if isinstance(data_list, (list, tuple)) else [data_list]
                        actual_len = len(base_list)
                        if actual_len == expected_len:
                            return base_list
                        elif actual_len < expected_len:
                            return base_list + ([pad_value] * (expected_len - actual_len))
                        else:
                            return base_list[:expected_len]

                    sim_noi = _pad_list(audit_sim.get("noi"), hold_period_audit)
                    sim_capex = _pad_list(audit_sim.get("capex"), hold_period_audit)
                    sim_interest = _pad_list(audit_sim.get("interest"), hold_period_audit)
                    sim_principal = _pad_list(audit_sim.get("principal"), hold_period_audit)

                    cfbds_row_vals_sim = [(n - c) if pd.notna(n) and pd.notna(c) else np.nan for n, c in zip(sim_noi, sim_capex)]
                    total_debt_service_vals_sim = [-(abs(i) + abs(p)) if pd.notna(i) and pd.notna(p) else np.nan for i, p in zip(sim_interest, sim_principal)]

                    audit_rows = [
                        ("--- Unlevered Cash Flows ---", [np.nan] * total_years),
                        ("Purchase Price", [-purchase_price] + [np.nan] * hold_period_audit),
                        ("Potential Gross Rent (PGR)", _pad_list([np.nan] + audit_sim.get("potential_rent", []), total_years)),
                        ("Less: Vacancy Loss", _pad_list([np.nan] + [-abs(v) if pd.notna(v) else np.nan for v in audit_sim.get("vacancy_loss", [])], total_years)),
                        ("Effective Gross Rent (EGR)", _pad_list([np.nan] + audit_sim.get("egr", []), total_years)),
                        ("Plus: Other Income", _pad_list([np.nan] + audit_sim.get("other_income", []), total_years)),
                        ("Effective Gross Income (EGI)", _pad_list([np.nan] + audit_sim.get("egi", []), total_years)),
                        ("Less: Operating Expenses", _pad_list([np.nan] + [-abs(e) if pd.notna(e) else np.nan for e in audit_sim.get("expenses", [])], total_years)),
                        ("Net Operating Income (NOI)", _pad_list([np.nan] + sim_noi, total_years)),
                        ("Less: Capital Expenditures (CapEx)", _pad_list([np.nan] + [-abs(c) if pd.notna(c) else np.nan for c in sim_capex], total_years)),
                        ("Cash Flow Before Debt Service (CFBDS)", [np.nan] + cfbds_row_vals_sim),
                        ("--- Levered Cash Flows & Debt ---", [np.nan] * total_years),
                        ("Net Loan Proceeds", initial_proceeds_net_row),
                        ("Net Refinancing Proceeds", refi_proceeds_net_row),
                        ("Loan Payoff", final_payoff_row),
                        ("Interest Paid", _pad_list([np.nan] + [-abs(i) if pd.notna(i) else np.nan for i in sim_interest], total_years)),
                        ("Principal Paid", _pad_list([np.nan] + [-abs(p) if pd.notna(p) else np.nan for p in sim_principal], total_years)),
                        ("Total Debt Service", [np.nan] + total_debt_service_vals_sim),
                        ("--- Sale ---", [np.nan] * total_years),
                        ("Net Sale Proceeds", [np.nan] * hold_period_audit + [net_sale_proceeds]),
                        ("--- Cash Flows for IRR ---", [np.nan] * total_years),
                        ("Unlevered Cash Flow (IRR)", _pad_list([-purchase_price] + audit_sim.get("unlevered_cf", [])[:-1] + [final_unlev_cf_for_irr], total_years)),
                        ("Levered Cash Flow (IRR)", _pad_list([-initial_equity] + audit_sim.get("levered_cf", [])[:-1] + [final_lev_cf_for_irr], total_years)),
                        ("--- Other Info ---", [np.nan] * total_years),
                        ("End of Year Loan Balance", _pad_list([initial_loan_proceeds] + audit_sim.get("loan_balance", []), total_years))
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
                            if abs(num_val) < 1e-3:
                                return "$0"
                            fmt_str = "${:,.0f}" if abs(num_val) >= 1 else "${:,.2f}"
                            return fmt_str.format(num_val) if num_val >= 0 else f"({fmt_str.format(abs(num_val))})"
                        except (TypeError, ValueError):
                            return str(val)
                    st.dataframe(
                        audit_df.style.format(fmt_audit, na_rep="-")
                        .set_properties(**{"text-align": "right"})
                        .set_table_styles([{"selector": "th", "props": [("text-align", "left")]}]),
                        use_container_width=True
                    )

                    st.download_button(
                        label=f"Download Sim #{selected_sim_index+1} Data (CSV)",
                        data=audit_df.to_csv().encode('utf-8'),
                        file_name=f"simulation_{selected_sim_index+1}_audit.csv",
                        mime="text/csv",
                        key=f"download_audit_{selected_sim_index}"
                    )

                    st.markdown("##### Returns for Selected Simulation (#{})".format(selected_sim_index + 1))

                    unlevered_irr_val = audit_sim.get('unlevered_irr', np.nan)
                    levered_irr_val = audit_sim.get('levered_irr', np.nan)
                    unlevered_mult_val = audit_sim.get('unlevered_multiple', np.nan)
                    levered_mult_val = audit_sim.get('levered_multiple', np.nan)

                    def format_multiple_audit(value):
                        return f"{value:.1f}x" if value is not None and np.isfinite(value) else "N/A"

                    col1, col2 = st.columns(2)
                    col1.metric("Unlevered IRR", f"{unlevered_irr_val:.1%}" if np.isfinite(unlevered_irr_val) else "N/A")
                    col2.metric("Levered IRR", f"{levered_irr_val:.1%}" if np.isfinite(levered_irr_val) else "N/A")

                    col3, col4 = st.columns(2)
                    col3.metric("Unlevered Multiple", format_multiple_audit(unlevered_mult_val), help="Using revised calc: Sum(+CFs)/(PP+Sum(|-CFs|))")
                    col4.metric("Levered Multiple", format_multiple_audit(levered_mult_val), help="Using revised calc: Sum(+CFs)/(Equity+Sum(|-CFs|))")

                    st.markdown("---")
                    st.subheader("Key Assumptions & Metrics (Simulation #{})".format(selected_sim_index + 1))
                    exit_cap_rate = audit_sim.get("sim_exit_cap_rate", np.nan)
                    assumption_rows_data = {
                        "Market Rent ($/Unit/Mo)": audit_sim.get("rent_per_unit", []),
                        "Vacancy Rate (%)": audit_sim.get("vacancy_rate", []),
                        "Rent Growth (%)": audit_sim.get("rent_growth_pct", []),
                        "Expense Growth (%)": audit_sim.get("expense_growth_pct", []),
                        "CapEx Growth (%)": audit_sim.get("capex_growth_pct", []),
                        ("Interest Rate (%)" if not inputs_used_for_run.is_variable_rate else "Effective Interest Rate (%)"): audit_sim.get("interest_rates", [])
                    }
                    assumption_rows = [(label, _pad_list(data, hold_period_audit)) for label, data in assumption_rows_data.items()]
                    if np.isfinite(exit_cap_rate):
                        cap_rate_row = [np.nan] * (hold_period_audit - 1) + [exit_cap_rate]
                        assumption_rows.append(("Exit Cap Rate (%)", cap_rate_row))
                    clean_assumptions = [(label, vals) for label, vals in assumption_rows if isinstance(vals, list) and len(vals) == hold_period_audit]
                    if clean_assumptions:
                        assumption_df = pd.DataFrame({label: vals for label, vals in clean_assumptions}).T
                        assumption_df.columns = [f"Yr {i+1}" for i in range(hold_period_audit)]
                        assumption_df.index.name = "Assumption"
                        def format_assumption_value(val, label):
                            if pd.isna(val):
                                return "-"
                            try:
                                f_val = float(val)
                                if "Rent ($/Unit/Mo)" in label:
                                    return f"${f_val:,.0f}"
                                elif ("Rate (%)" in label and "Growth" not in label):
                                    return f"{f_val:.1%}"
                                elif "Growth (%)" in label:
                                    return f"{f_val:.1f}%"
                                else:
                                    return f"{f_val:.2f}"
                            except (ValueError, TypeError):
                                return str(val)
                        formatted_df = assumption_df.apply(lambda row: pd.Series([format_assumption_value(val, row.name) for val in row], index=row.index), axis=1)
                        st.dataframe(formatted_df.style.set_properties(**{"text-align": "right"}).set_table_styles([{"selector": "th", "props": [("text-align", "left")]}]), use_container_width=True)
                    else:
                        st.warning("Could not display key assumptions for this simulation.")

        with tabs[tab_keys.index("🚪 Exit")]:
            st.subheader("Exit Analysis")
            st.caption("Distribution of Net Sale Proceeds & Simulated Exit Cap Rates across all valid runs.")

            # Validate input data
            valid_exit_values = [x for x in finite_exit_values if np.isfinite(x)] if finite_exit_values else []
            valid_exit_caps = [x if x <= 1.0 else x / 100.0 for x in finite_exit_caps if np.isfinite(x)] if finite_exit_caps else []
            
            if not valid_exit_values or not valid_exit_caps or len(set(valid_exit_values)) < 2 or len(set(valid_exit_caps)) < 2:
                st.warning("Not enough valid or varied exit results for analysis. Need at least two distinct finite values.")
            else:
                # Calculate metrics locally
                mean_exit_val = np.mean(valid_exit_values) if valid_exit_values else np.nan
                median_exit_val = np.median(valid_exit_values) if valid_exit_values else np.nan
                p05_exit_val = np.percentile(valid_exit_values, 5) if valid_exit_values else np.nan
                p95_exit_val = np.percentile(valid_exit_values, 95) if valid_exit_values else np.nan
                mean_exit_cap = np.mean(valid_exit_caps) if valid_exit_caps else np.nan
                median_exit_cap = np.median(valid_exit_caps) if valid_exit_caps else np.nan
                p05_cap = np.percentile(valid_exit_caps, 5) if valid_exit_caps else np.nan
                p95_cap = np.percentile(valid_exit_caps, 95) if valid_exit_caps else np.nan

                # Net Exit Value Section
                st.markdown("#### Net Exit Value (After Transaction Costs)")
                col1_exit, col2_exit, col3_exit, col4_exit = st.columns(4)
                col1_exit.metric("Mean", f"${mean_exit_val:,.0f}" if np.isfinite(mean_exit_val) else "N/A")
                col2_exit.metric("Median", f"${median_exit_val:,.0f}" if np.isfinite(median_exit_val) else "N/A")
                col3_exit.metric("5th Pctl", f"${p05_exit_val:,.0f}" if np.isfinite(p05_exit_val) else "N/A")
                col4_exit.metric("95th Pctl", f"${p95_exit_val:,.0f}" if np.isfinite(p95_exit_val) else "N/A")
                try:
                    # Calculate histogram
                    min_val = min(valid_exit_values)
                    max_val = max(valid_exit_values)
                    spread = max_val - min_val
                    x_range = (min_val - spread * 0.1, max_val + spread * 0.1) if spread > 1e-6 else (min_val - 1e6, max_val + 1e6)
                    hist_vals, bin_edges = np.histogram(valid_exit_values, bins=30, range=x_range)
                    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
                    total_count = sum(hist_vals)
                    percentages = (hist_vals / total_count * 100.0) if total_count > 0 else np.zeros_like(hist_vals)
                    max_y = max(percentages) * 1.20 if any(p > 0 for p in percentages) else 1.0
                    annotation_y_offset = max_y * 0.08

                    fig_exit_val = go.Figure(data=[go.Bar(x=bin_centers, y=percentages, marker_color='mediumseagreen', opacity=0.8, name="Frequency")])

                    # Add markers with annotations above vertical lines
                    if np.isfinite(mean_exit_val):
                        fig_exit_val.add_shape(
                            type="line",
                            x0=mean_exit_val, x1=mean_exit_val,
                            y0=0, y1=max_y * 0.90,
                            line=dict(color="red", dash="dash", width=1.5)
                        )
                        fig_exit_val.add_annotation(
                            x=mean_exit_val,
                            y=max_y * 1.05,
                            text=f"Mean: ${mean_exit_val:,.0f}",
                            showarrow=False,
                            font=dict(color="red"),
                            yshift=annotation_y_offset * 2,
                            yanchor="bottom"
                        )
                    if np.isfinite(median_exit_val):
                        fig_exit_val.add_shape(
                            type="line",
                            x0=median_exit_val, x1=median_exit_val,
                            y0=0, y1=max_y * 0.90,
                            line=dict(color="purple", dash="dash", width=1.5)
                        )
                        fig_exit_val.add_annotation(
                            x=median_exit_val,
                            y=max_y * 1.05,
                            text=f"Median: ${median_exit_val:,.0f}",
                            showarrow=False,
                            font=dict(color="purple"),
                            yshift=-annotation_y_offset * 12,
                            yanchor="bottom"
                        )
                    if np.isfinite(p05_exit_val):
                        fig_exit_val.add_shape(
                            type="line",
                            x0=p05_exit_val, x1=p05_exit_val,
                            y0=0, y1=max_y * 0.90,
                            line=dict(color="darkgrey", dash="dot", width=1.5)
                        )
                        fig_exit_val.add_annotation(
                            x=p05_exit_val,
                            y=max_y * 1.05,
                            text=f"5th: ${p05_exit_val:,.0f}",
                            showarrow=False,
                            font=dict(color="darkgrey", size=10),
                            yshift=annotation_y_offset * 1.5,
                            yanchor="bottom"
                        )
                    if np.isfinite(p95_exit_val):
                        fig_exit_val.add_shape(
                            type="line",
                            x0=p95_exit_val, x1=p95_exit_val,
                            y0=0, y1=max_y * 0.90,
                            line=dict(color="darkgrey", dash="dot", width=1.5)
                        )
                        fig_exit_val.add_annotation(
                            x=p95_exit_val,
                            y=max_y * 1.05,
                            text=f"95th: ${p95_exit_val:,.0f}",
                            showarrow=False,
                            font=dict(color="darkgrey", size=10),
                            yshift=annotation_y_offset * 3,
                            yanchor="bottom"
                        )

                    fig_exit_val.update_layout(
                        title=f"Distribution of Net Exit Values ({len(valid_exit_values)} runs)",
                        xaxis_title="Net Exit Value ($)",
                        yaxis_title="% of Instances",
                        bargap=0.1,
                        showlegend=False,
                        template="plotly_white",
                        yaxis_ticksuffix="%"
                    )
                    fig_exit_val.update_xaxes(tickformat="$,.0s", range=x_range)
                    st.plotly_chart(fig_exit_val, use_container_width=True)
                    st.caption(
                        "Shows the frequency distribution of potential Net Sale Proceeds (after transaction costs). "
                        "The spread reflects uncertainty driven primarily by exit year NOI and simulated Exit Cap Rate volatility."
                    )
                except Exception as e:
                    logger.error(f"Error plotting exit value distribution: {e}, values={valid_exit_values[:5]}, metrics={mean_exit_val, median_exit_val, p05_exit_val, p95_exit_val}, range={x_range}")
                    st.warning("Could not plot exit value distribution.")

                st.markdown("---")
                st.markdown("#### Simulated Exit Cap Rate")
                col1_cap, col2_cap, col3_cap, col4_cap = st.columns(4)
                col1_cap.metric("Mean", f"{mean_exit_cap*100:.2f}%" if np.isfinite(mean_exit_cap) else "N/A")
                col2_cap.metric("Median", f"{median_exit_cap*100:.2f}%" if np.isfinite(median_exit_cap) else "N/A")
                col3_cap.metric("5th Pctl", f"{p05_cap*100:.2f}%" if np.isfinite(p05_cap) else "N/A")
                col4_cap.metric("95th Pctl", f"{p95_cap*100:.2f}%" if np.isfinite(p95_cap) else "N/A")
                try:
                    # Calculate histogram with x-axis in percentage
                    min_cap = min(valid_exit_caps)
                    max_cap = max(valid_exit_caps)
                    spread_cap = max_cap - min_cap
                    x_range_cap = (min_cap - spread_cap * 0.1, max_cap + spread_cap * 0.1) if spread_cap > 1e-6 else (min_cap - 0.01, max_cap + 0.01)
                    hist_caps, bin_edges_caps = np.histogram(valid_exit_caps, bins=30, range=x_range_cap)
                    bin_centers_caps = [x * 100 for x in (0.5 * (bin_edges_caps[1:] + bin_edges_caps[:-1]))]  # Scale to percentage
                    total_count_caps = sum(hist_caps)
                    percentages_caps = (hist_caps / total_count_caps * 100.0) if total_count_caps > 0 else np.zeros_like(hist_caps)
                    max_y_caps = max(percentages_caps) * 1.20 if any(p > 0 for p in percentages_caps) else 1.0
                    annotation_y_offset_caps = max_y_caps * 0.08

                    fig_exit_cap = go.Figure(data=[go.Bar(x=bin_centers_caps, y=percentages_caps, marker_color='mediumpurple', opacity=0.8, name="Frequency")])

                    # Add markers with annotations above vertical lines
                    if np.isfinite(mean_exit_cap):
                        fig_exit_cap.add_shape(
                            type="line",
                            x0=mean_exit_cap * 100, x1=mean_exit_cap * 100,  # Scale to percentage
                            y0=0, y1=max_y_caps * 0.90,
                            line=dict(color="red", dash="dash", width=1.5)
                        )
                        fig_exit_cap.add_annotation(
                            x=mean_exit_cap * 100,
                            y=max_y_caps * 1.05,
                            text=f"Mean: {mean_exit_cap*100:.2f}%",
                            showarrow=False,
                            font=dict(color="red"),
                            yshift=annotation_y_offset_caps * 2,
                            yanchor="bottom"
                        )
                    if np.isfinite(median_exit_cap):
                        fig_exit_cap.add_shape(
                            type="line",
                            x0=median_exit_cap * 100, x1=median_exit_cap * 100,
                            y0=0, y1=max_y_caps * 0.90,
                            line=dict(color="purple", dash="dash", width=1.5)
                        )
                        fig_exit_cap.add_annotation(
                            x=median_exit_cap * 100,
                            y=max_y_caps * 1.05,
                            text=f"Median: {median_exit_cap*100:.2f}%",
                            showarrow=False,
                            font=dict(color="purple"),
                            yshift=-annotation_y_offset_caps * 12,
                            yanchor="bottom"
                        )
                    if np.isfinite(p05_cap):
                        fig_exit_cap.add_shape(
                            type="line",
                            x0=p05_cap * 100, x1=p05_cap * 100,
                            y0=0, y1=max_y_caps * 0.90,
                            line=dict(color="darkgrey", dash="dot", width=1.5)
                        )
                        fig_exit_cap.add_annotation(
                            x=p05_cap * 100,
                            y=max_y_caps * 1.05,
                            text=f"5th: {p05_cap*100:.2f}%",
                            showarrow=False,
                            font=dict(color="darkgrey", size=10),
                            yshift=annotation_y_offset_caps * 1.5,
                            yanchor="bottom"
                        )
                    if np.isfinite(p95_cap):
                        fig_exit_cap.add_shape(
                            type="line",
                            x0=p95_cap * 100, x1=p95_cap * 100,
                            y0=0, y1=max_y_caps * 0.90,
                            line=dict(color="darkgrey", dash="dot", width=1.5)
                        )
                        fig_exit_cap.add_annotation(
                            x=p95_cap * 100,
                            y=max_y_caps * 1.05,
                            text=f"95th: {p95_cap*100:.2f}%",
                            showarrow=False,
                            font=dict(color="darkgrey", size=10),
                            yshift=annotation_y_offset_caps * 3,
                            yanchor="bottom"
                        )

                    fig_exit_cap.update_layout(
                        title=f"Distribution of Simulated Exit Cap Rates ({len(valid_exit_caps)} runs)",
                        xaxis_title="Exit Cap Rate (%)",
                        yaxis_title="% of Instances",
                        bargap=0.1,
                        showlegend=False,
                        template="plotly_white",
                        yaxis_ticksuffix="%"
                    )
                    fig_exit_cap.update_xaxes(ticksuffix="%", tickformat=".1f", range=[x * 100 for x in x_range_cap])
                    st.plotly_chart(fig_exit_cap, use_container_width=True)
                    st.caption(
                        "Shows the frequency distribution of potential Exit Cap Rates at sale. This is influenced by the 'Average Exit Cap Rate' input, "
                        "its associated volatility, and the 'Exit Cap Adj. Factor' based on terminal year rent growth."
                    )
                except Exception as e:
                    logger.error(f"Error plotting exit cap distribution: {e}, values={valid_exit_caps[:5]}, metrics={mean_exit_cap, median_exit_cap, p05_cap, p95_cap}, range={x_range_cap}")
                    st.warning("Could not plot exit cap rate distribution.")

        with tabs[tab_keys.index("🔎 Sensitivity")]:
            st.subheader("Sensitivity Analysis")
            st.markdown("Analyze how key inputs affect the Mean Levered IRR based on the *last run simulation* results' baseline.")
            st.caption("Test multiple parameters. Each parameter will be shown in a separate chart.")
            param_options = {
                "Purchase Price ($)": ("purchase_price", "currency"),
                "Initial Market Rent ($/Mo)": ("base_rent", "currency"),
                "Initial Market Rent Premium/Discount to Fair Value (%)": ("market_rent_deviation_pct", "percent_decimal"),
                "Years to Normalize Rent": ("market_convergence_years", "integer"),
                "Avg Fair Value Rent Growth (Normal, %)": ("normal_growth_mean", "percent_direct"),
                "Fair Value Rent Growth Vol (Normal, %)": ("normal_growth_vol", "percent_direct"),
                "Prob. Normal → Recession (%)": ("transition_normal_to_recession", "percent_decimal"),
                "Avg Fair Value Rent Growth (Recession, %)": ("recession_growth_mean", "percent_direct"),
                "Fair Value Rent Growth Vol (Recession, %)": ("recession_growth_vol", "percent_direct"),
                "Prob. Recession → Normal (%)": ("transition_recession_to_normal", "percent_decimal"),
                "Starting Vacancy Rate (%)": ("current_vacancy", "percent_decimal"),
                "Target Long-Term Vacancy (%)": ("stabilized_vacancy", "percent_decimal"),
                "Vacancy Reversion Speed": ("vacancy_reversion_speed", "decimal_places_2"),
                "Vacancy Rate Volatility (% pts/Yr)": ("vacancy_volatility", "percent_decimal"),
                "Initial Annual Other Income ($)": ("mean_other_income", "currency"),
                "Avg Other Income Growth (%)": ("mean_other_income_growth", "percent_direct"),
                "Other Income Growth Vol (%)": ("other_income_stddev", "percent_direct"),
                "Initial Annual OpEx ($)": ("mean_expense", "currency"),
                "Average OpEx Growth (%)": ("mean_expense_growth", "percent_direct"),
                "OpEx Growth Volatility (%)": ("expense_stddev", "percent_direct"),
                "Initial CapEx Reserve ($/Unit/Yr)": ("capex_per_unit_yr", "currency"),
                "Average CapEx Growth (%)": ("mean_capex_growth", "percent_direct"),
                "CapEx Growth Volatility (%)": ("capex_stddev", "percent_direct"),
                "Average Exit Cap Rate (%)": ("mean_exit_cap_rate", "percent_direct"),
                "Exit Cap Rate Volatility (% pts)": ("exit_cap_rate_stddev", "percent_direct"),
                "Transaction Cost on Sale (%)": ("transaction_cost_pct", "percent_decimal"),
                "Exit Cap Adj. for Rent Growth": ("exit_cap_rent_growth_sensitivity", "decimal_places_2"),
                "Gross Exit Value Floor ($)": ("exit_floor_value", "currency"),
                "Loan-to-Cost Ratio (%)": ("loan_to_cost", "percent_decimal"),
                "Fixed Loan Interest Rate (%)": ("interest_rate", "percent_decimal"),
                "Spread Over SOFR (%)": ("sofr_spread", "percent_decimal"),
                "SOFR Floor (%)": ("sofr_floor", "percent_decimal"),
                "Floating Rate Volatility Factor": ("volatility_scalar", "decimal_places_1"),
                "Cyclical Persistence (Rates)": ("rate_persistence_phi", "decimal_places_2"),
                "Amortization Period (Years)": ("loan_term_yrs", "integer"),
                "Corr: Rent & Expense Shocks": ("corr_rent_expense", "decimal_places_2"),
                "Corr: Rent & Other Income Shocks": ("corr_rent_other_income", "decimal_places_2"),
                "Corr: Rent & Vacancy Shocks": ("corr_rent_vacancy", "decimal_places_2"),
                "Risk-Free Rate (%)": ("risk_free_rate", "percent_decimal"),
                "Hurdle Rate (% IRR)": ("hurdle_rate", "percent_decimal"),
                "Hold Period (Years)": ("hold_period", "integer"),
                "Cyclical Persistence (Growth)": ("growth_persistence_phi", "decimal_places_2"),
            }
            if "inputs" not in st.session_state or not st.session_state["inputs"]:
                st.warning("Inputs not found. Run simulation first.")
            else:
                selected_params_display = st.multiselect("Select Parameters to Vary (1–5)", options=list(param_options.keys()), max_selections=5, key="multi_sens_params")
                col_step, col_sims = st.columns(2)
                num_steps = col_step.slider("Number of Steps per Parameter", min_value=3, max_value=11, value=5, step=2, key="sens_steps")
                sens_num_sims = col_sims.number_input("Simulations per Step", min_value=10, max_value=5000, value=200, step=50, key="sens_num_sims")
                st.markdown("---")
                st.markdown("###### Define Sensitivity Range")
                param_ranges = {}  # Initialize param_ranges
                rel_variation_pct = st.slider(f"Relative Variation (+/- % of baseline)", min_value=1.0, max_value=50.0, value=10.0, step=1.0, key="sens_rel_pct", format="%.0f%%") / 100.0
                for param_display_name in selected_params_display:
                    param_ranges[param_display_name] = {"type": "relative", "value": rel_variation_pct}

                st.markdown("---")
                run_button_sens = st.button("Run Sensitivity Analysis", key="run_multi_sensitivity")
                if run_button_sens and selected_params_display:
                    base_inputs_sens = st.session_state['inputs'].copy()
                    all_sensitivity_results = []
                    baseline_irr = st.session_state.get("processed_results", {}).get("metrics", {}).get("mean_levered_irr", np.nan)
                    for param_display_name in selected_params_display:
                        param_key, unit_type = param_options[param_display_name]
                        base_value = base_inputs_sens.get(param_key)
                        range_settings = param_ranges.get(param_display_name)
                        if base_value is None or range_settings is None:
                            st.warning(f"Skipping '{param_display_name}': Missing baseline or range.")
                            continue
                        test_values_internal = []
                        test_values_display = []
                        if range_settings["type"] == "relative":
                            rel_pct = range_settings["value"]
                            min_val_internal = base_value * (1.0 - rel_pct)
                            max_val_internal = base_value * (1.0 + rel_pct)
                            if np.isclose(base_value, 0):
                                if unit_type == "percent_decimal":
                                    default_abs_var = 0.01
                                elif unit_type == "percent_direct":
                                    default_abs_var = 1.0
                                elif unit_type == "decimal_places_2":
                                    default_abs_var = 0.05
                                else:
                                    default_abs_var = 0.1
                                min_val_internal = base_value - default_abs_var
                                max_val_internal = base_value + default_abs_var
                            test_values_internal = np.linspace(min_val_internal, max_val_internal, num_steps)
                        if unit_type == "percent_decimal":
                            test_values_display = [v * 100.0 for v in test_values_internal]
                            base_value_display = base_value * 100.0
                            x_suffix = " (%)"
                            x_format = "%.1f%%"
                        elif unit_type == "percent_direct":
                            test_values_display = test_values_internal
                            base_value_display = base_value
                            x_suffix = " (%)"
                            x_format = "%.1f%%"
                        elif unit_type == "currency":
                            test_values_display = test_values_internal
                            base_value_display = base_value
                            x_suffix = " ($)"
                            x_format = "$%.0f"
                        elif unit_type == "integer":
                            test_values_display = np.round(test_values_internal).astype(int)
                            base_value_display = int(round(base_value))
                            x_suffix = ""
                            x_format = "%d"
                        elif "decimal_places" in unit_type:
                            test_values_display = test_values_internal
                            base_value_display = base_value
                            places = int(unit_type.split('_')[-1])
                            x_suffix = ""
                            x_format = f"%.{places}f"
                        else:
                            test_values_display = test_values_internal
                            base_value_display = base_value
                            x_suffix = ""
                            x_format = "%.2f"
                        irrs_for_param = []
                        status_msg = f"Running sensitivity for {param_display_name}..."
                        with st.status(status_msg, expanded=True) as status:
                            for i, internal_val in enumerate(test_values_internal):
                                display_val_str = x_format % test_values_display[i]
                                if '%' in display_val_str:
                                    display_val_str = display_val_str.replace('%', '') + '%'
                                st.write(f"Step {i+1}/{num_steps}: Testing {param_key} = {display_val_str}...")
                                temp_inputs_dict = base_inputs_sens.copy()
                                temp_inputs_dict[param_key] = internal_val
                                try:
                                    temp_sim_inputs = SimulationInputs(**temp_inputs_dict)
                                    result = run_monte_carlo(temp_sim_inputs, sens_num_sims, forward_curve_data, std_dev_curve_data)
                                    irr = result["metrics"].get("mean_levered_irr", np.nan)
                                    irrs_for_param.append(irr)
                                    st.write(f"-> Mean Levered IRR: {irr:.1%}" if np.isfinite(irr) else "-> Mean Levered IRR: N/A")
                                except Exception as e:
                                    irrs_for_param.append(np.nan)
                                    st.write(f"-> Error running simulation: {e}")
                                    logger.error(f"Sensitivity failed for {param_key}={internal_val}: {e}")
                            status.update(label=f"Sensitivity analysis for {param_display_name} complete!", state="complete", expanded=False)
                        for disp_val, irr in zip(test_values_display, irrs_for_param):
                            all_sensitivity_results.append({"Parameter": param_display_name, "Value_Display": disp_val, "Mean Levered IRR": irr * 100.0 if np.isfinite(irr) else None})
                        fig_sens = go.Figure()
                        valid_indices = [i for i, irr in enumerate(irrs_for_param) if np.isfinite(irr)]
                        plot_x = [test_values_display[i] for i in valid_indices]
                        plot_y = [irrs_for_param[i] * 100 for i in valid_indices]
                        if plot_x and plot_y:
                            fig_sens.add_trace(go.Scatter(x=plot_x, y=plot_y, mode="lines+markers", name=param_display_name))
                            if np.isfinite(baseline_irr):
                                closest_x_to_baseline = min(test_values_display, key=lambda x: abs(x - base_value_display))
                                fig_sens.add_trace(go.Scatter(
                                    x=[base_value_display],
                                    y=[baseline_irr * 100],
                                    mode="markers",
                                    marker=dict(color="red", size=10, symbol="x"),
                                    name="Baseline"
                                ))
                            baseline_text_fmt = x_format % base_value_display
                            if '%' in baseline_text_fmt:
                                baseline_text_fmt = baseline_text_fmt.replace('%', '') + '%'
                            baseline_text = f"Baseline: {baseline_text_fmt}"
                            fig_sens.add_vline(
                                x=base_value_display,
                                line_dash="dash",
                                line_color="rgba(255,0,0,0.5)",
                                annotation_text=baseline_text,
                                annotation_position="bottom right"
                            )
                            fig_sens.update_layout(
                                title=f"Mean Levered IRR Sensitivity to {param_display_name}",
                                xaxis_title=f"{param_display_name}{x_suffix}",
                                yaxis_title="Mean Levered IRR (%)",
                                yaxis_tickformat=".1f",
                                hovermode="x unified",
                                template="plotly_white",
                                showlegend=True,
                                legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
                            )
                            if unit_type == "currency":
                                fig_sens.update_xaxes(tickformat="$,.0f")
                            elif '%' in x_suffix:
                                fig_sens.update_xaxes(ticksuffix="%")
                            st.plotly_chart(fig_sens, use_container_width=True)
                        else:
                            st.warning(f"No valid IRR results for {param_display_name}.")
                        st.markdown("---")
                    if all_sensitivity_results:
                        csv_df = pd.DataFrame(all_sensitivity_results)
                        csv_bytes = csv_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="📥 Download Full Sensitivity Results as CSV",
                            data=csv_bytes,
                            file_name="sensitivity_analysis_results.csv",
                            mime="text/csv"
                        )

        with tabs[tab_keys.index("🗂️ Scenarios")]:
            st.subheader("Scenario Management & Comparison")
            st.markdown("### Save Current Scenario Results")
            scenario_name = st.text_input("Scenario Name", key="save_scenario_name_mgmt", help="Enter a unique name for this scenario run.")
            scenario_desc = st.text_area("Optional Notes / Tags", key="save_scenario_desc_mgmt", help="Add notes about assumptions or purpose.")
            if st.button("📎 Save Current Scenario & Results", key="save_button_mgmt", use_container_width=True):
                if scenario_name.strip():
                    saved_result = st.session_state.get("processed_results")
                    if saved_result and isinstance(saved_result, dict) and "error" not in saved_result:
                        inputs_copy = st.session_state["inputs"].copy()
                        results_copy = saved_result.copy()
                        st.session_state["saved_scenarios"][scenario_name] = {
                            "inputs": inputs_copy,
                            "results": results_copy,
                            "description": scenario_desc,
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        }
                        st.success(f"Saved scenario '{scenario_name}'.")
                    else:
                        st.warning("No valid results to save.")
                else:
                    st.warning("Please enter a scenario name.")
            st.markdown("---")
            st.markdown("### Saved Scenarios")
            if not st.session_state.get("saved_scenarios"):
                st.info("No scenarios saved yet.")
            else:
                sorted_scenario_keys = sorted(st.session_state["saved_scenarios"].keys())
                for name in sorted_scenario_keys:
                    if name in st.session_state["saved_scenarios"]:
                        details = st.session_state["saved_scenarios"][name]
                        with st.expander(f"{name} — Saved: {details.get('timestamp', 'N/A')}"):
                            st.caption(f"Notes: {details.get('description', '_No description provided._')}")
                            col1, col2, col3 = st.columns([1, 1, 2])
                            if col1.button("🔄 Load Inputs", key=f"load_{name}", help="Load inputs from this scenario into the sidebar."):
                                st.session_state["inputs"] = details["inputs"].copy()
                                st.session_state["processed_results"] = None
                                st.success(f"Inputs from '{name}' loaded.")
                                time.sleep(1)
                                st.rerun()
                            if col2.button("🗑️ Delete", key=f"delete_{name}", help="Delete this saved scenario."):
                                del st.session_state["saved_scenarios"][name]
                                st.rerun()
                            new_name = col3.text_input(f"Rename '{name}' to:", value=name, key=f"rename_{name}", label_visibility="collapsed")
                            rename_button_key = f"rename_btn_{name}"
                            if new_name != name and new_name.strip():
                                if new_name in st.session_state["saved_scenarios"]:
                                    col3.warning("Name exists.")
                                elif col3.button("✏️ Rename", key=rename_button_key):
                                    st.session_state["saved_scenarios"][new_name] = st.session_state["saved_scenarios"].pop(name)
                                    st.rerun()
            st.markdown("---")
            st.markdown("### Compare Two Saved Scenarios")
            if len(st.session_state.get("saved_scenarios", {})) < 2:
                st.info("Save at least two scenarios.")
            else:
                scenario_keys = sorted(list(st.session_state["saved_scenarios"].keys()))
                col_a, col_b = st.columns(2)
                scenario_a = col_a.selectbox("Select Scenario A", options=scenario_keys, index=0, key="comp_a")
                default_b_index = 1 if len(scenario_keys) > 1 else 0
                if scenario_a == scenario_keys[default_b_index] and len(scenario_keys) > 1:
                    default_b_index = 0
                scenario_b = col_b.selectbox("Select Scenario B", options=scenario_keys, index=default_b_index, key="comp_b")
                if scenario_a and scenario_b and scenario_a != scenario_b:
                    if scenario_a not in st.session_state["saved_scenarios"] or scenario_b not in st.session_state["saved_scenarios"]:
                        st.warning("Scenario not found.")
                    else:
                        data_a = st.session_state["saved_scenarios"][scenario_a]
                        data_b = st.session_state["saved_scenarios"][scenario_b]
                        result_a = data_a.get("results", {})
                        result_b = data_b.get("results", {})
                        inputs_a = data_a.get("inputs", {})
                        inputs_b = data_b.get("inputs", {})
                        st.markdown("#### 🔍 Scenario Comparison: Key Metrics")
                        def get_metric_and_delta(metric_key_path, res_a, res_b):
                            val_a = res_a
                            val_b = res_b
                            try:
                                for key in metric_key_path:
                                    val_a = val_a.get(key, {})
                                    val_b = val_b.get(key, {})
                                val_a = val_a if isinstance(val_a, (int, float, np.number)) else None
                                val_b = val_b if isinstance(val_b, (int, float, np.number)) else None
                                delta = None
                                if val_a is not None and val_b is not None and np.isfinite(val_a) and np.isfinite(val_b):
                                    delta = val_b - val_a
                                return val_a, val_b, delta
                            except Exception as e:
                                logger.error(f"Error get_metric_delta {metric_key_path}: {e}")
                                return None, None, None
                        metrics_to_show_comp = [
                            ("Mean Levered IRR", ("metrics", "mean_levered_irr"), lambda x: f"{x:.1%}" if x is not None and np.isfinite(x) else "N/A", lambda d: f"{d*100:.1f} pts" if d is not None else ""),
                            ("Median Levered IRR", ("metrics", "median_levered_irr"), lambda x: f"{x:.1%}" if x is not None and np.isfinite(x) else "N/A", lambda d: f"{d*100:.1f} pts" if d is not None else ""),
                            ("Mean Unlevered IRR", ("metrics", "mean_unlevered_irr"), lambda x: f"{x:.1%}" if x is not None and np.isfinite(x) else "N/A", lambda d: f"{d*100:.1f} pts" if d is not None else ""),
                            ("Median Unlevered IRR", ("metrics", "median_unlevered_irr"), lambda x: f"{x:.1%}" if x is not None and np.isfinite(x) else "N/A", lambda d: f"{d*100:.1f} pts" if d is not None else ""),
                            ("Mean Net Exit Value", ("metrics", "mean_exit_value"), lambda x: f"${x:,.0f}" if x is not None and np.isfinite(x) else "N/A", lambda d: f"${d:,.0f}" if d is not None else ""),
                            ("Mean Exit Cap Rate", ("metrics", "mean_exit_cap"), lambda x: f"{x*100:.2f}%" if x is not None and np.isfinite(x) else "N/A", lambda d: f"{d*100:.2f} pts" if d is not None else ""),
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
                        st.markdown("---")
                        st.markdown("#### 📋 Assumptions Comparison")
                        def format_value_compare(k, v):
                            if v is None:
                                return "N/A"
                            dollar_keys = {"purchase_price", "mean_other_income", "mean_expense", "exit_floor_value", "base_rent", "capex_per_unit_yr"}
                            percent_decimal_keys = {"market_rent_deviation_pct", "transition_normal_to_recession", "transition_recession_to_normal", "current_vacancy", "stabilized_vacancy", "vacancy_volatility", "loan_to_cost", "interest_rate", "transaction_cost_pct", "risk_free_rate", "hurdle_rate", "sofr_spread", "sofr_floor"}
                            percent_direct_keys = {"normal_growth_mean", "normal_growth_vol", "recession_growth_mean", "recession_growth_vol", "mean_other_income_growth", "other_income_stddev", "mean_expense_growth", "expense_stddev", "mean_capex_growth", "capex_stddev", "mean_exit_cap_rate", "exit_cap_rate_stddev"}
                            float_2dp_keys = {"vacancy_reversion_speed", "growth_persistence_phi", "corr_rent_expense", "corr_rent_other_income", "corr_rent_vacancy", "exit_cap_rent_growth_sensitivity", "rate_persistence_phi"}
                            float_1dp_keys = {"volatility_scalar"}
                            int_keys = {"num_units", "market_convergence_years", "loan_term_yrs", "hold_period", "num_simulations"}
                            try:
                                if k in dollar_keys:
                                    return f"${float(v):,.0f}"
                                elif k in percent_decimal_keys:
                                    return f"{float(v)*100:.1f}%"
                                elif k in percent_direct_keys:
                                    return f"{float(v):.1f}%"
                                elif k in float_2dp_keys:
                                    return f"{float(v):.2f}"
                                elif k in float_1dp_keys:
                                    return f"{float(v):.1f}"
                                elif k in int_keys:
                                    return f"{int(v):d}"
                                elif k == "loan_type":
                                    return str(v)
                                elif k == "use_correlations":
                                    return "Yes" if v else "No"
                                elif k == "is_variable_rate":
                                    return "Floating" if v else "Fixed"
                                elif isinstance(v, (int, float, np.number)):
                                    return f"{v:.2f}"
                                else:
                                    return str(v)
                            except (ValueError, TypeError):
                                return str(v)
                        input_labels = {k: k.replace("_", " ").title() for k in SimulationInputs.__annotations__ if not k.startswith('_') and not isinstance(getattr(SimulationInputs, k, None), property)}
                        differences_summary = []
                        for k in sorted(input_labels.keys()):
                            if k in inputs_a or k in inputs_b:
                                val_a = inputs_a.get(k)
                                val_b = inputs_b.get(k)
                                fmt_a = format_value_compare(k, val_a)
                                fmt_b = format_value_compare(k, val_b)
                                if fmt_a != fmt_b:
                                    differences_summary.append(f"- **{input_labels[k]}**: {scenario_a}=`{fmt_a}`, {scenario_b}=`{fmt_b}`")
                        if differences_summary:
                            st.markdown("**Key Assumption Differences:**")
                            st.markdown("\n".join(differences_summary))
                            st.markdown("---")
                        else:
                            st.info("Inputs appear identical.")
                        rows_assumptions = []
                        for k in sorted(input_labels.keys()):
                            if k in inputs_a or k in inputs_b:
                                val_a = inputs_a.get(k)
                                val_b = inputs_b.get(k)
                                fmt_a = format_value_compare(k, val_a)
                                fmt_b = format_value_compare(k, val_b)
                                is_different = fmt_a != fmt_b
                                row = {"Assumption": input_labels[k], scenario_a: fmt_a, scenario_b: fmt_b, "IsDifferent": is_different}
                                rows_assumptions.append(row)
                        df_compare_assumptions = pd.DataFrame(rows_assumptions).set_index("Assumption")
                        show_only_diff = st.checkbox("Show only differing assumptions", key="scenario_diff_check")
                        df_display_assumptions = df_compare_assumptions[df_compare_assumptions["IsDifferent"]].drop(columns=["IsDifferent"]) if show_only_diff else df_compare_assumptions.drop(columns=["IsDifferent"])
                        if not df_display_assumptions.empty:
                            st.dataframe(df_display_assumptions, use_container_width=True)
                        elif show_only_diff:
                            st.info("No differing assumptions found.")
                        st.markdown("#### Levered IRR Distributions")
                        irrs_a = result_a.get("finite_levered_irrs", [])
                        irrs_b = result_b.get("finite_levered_irrs", [])
                        if not irrs_a and not irrs_b:
                            st.warning("No valid Levered IRR data for comparison plot.")
                        else:
                            combined_irrs = [irr for irr in irrs_a if np.isfinite(irr)] + [irr for irr in irrs_b if np.isfinite(irr)]
                            if not combined_irrs:
                                st.warning("No finite Levered IRR data found to plot.")
                            else:
                                global_min = np.min(combined_irrs)
                                global_max = np.max(combined_irrs)
                                padding = (global_max - global_min) * 0.05
                                if np.isclose(global_max, global_min):
                                    padding = 0.05
                                x_start = global_min - padding
                                x_end = global_max + padding
                                num_bins = 30
                                bin_size = (x_end - x_start) / num_bins if not np.isclose(x_end, x_start) else 0.1
                                fig_comp = make_subplots(
                                    rows=2,
                                    cols=1,
                                    shared_xaxes=True,
                                    vertical_spacing=0.05,
                                    subplot_titles=(f"Scenario: {scenario_a}", f"Scenario: {scenario_b}")
                                )
                                if bin_size > FLOAT_ATOL:
                                    fig_comp.add_trace(
                                        go.Histogram(
                                            x=irrs_a,
                                            name=scenario_a,
                                            marker_color='rgba(55, 128, 191, 0.7)',
                                            xbins=dict(start=x_start, end=x_end, size=bin_size),
                                            showlegend=False
                                        ),
                                        row=1,
                                        col=1
                                    )
                                    fig_comp.add_trace(
                                        go.Histogram(
                                            x=irrs_b,
                                            name=scenario_b,
                                            marker_color='rgba(255, 127, 14, 0.7)',
                                            xbins=dict(start=x_start, end=x_end, size=bin_size),
                                            showlegend=False
                                        ),
                                        row=2,
                                        col=1
                                    )
                                    fig_comp.update_layout(
                                        title_text="Levered IRR Distributions (Comparison)",
                                        height=600,
                                        template="plotly_white",
                                        bargap=0.1,
                                        margin=dict(t=60, b=10, l=10, r=10)
                                    )
                                    fig_comp.update_xaxes(tickformat=".1%", title_text="Levered IRR", row=2, col=1)
                                    fig_comp.update_yaxes(title_text="Frequency", row=1, col=1)
                                    fig_comp.update_yaxes(title_text="Frequency", row=2, col=1)
                                    st.plotly_chart(fig_comp, use_container_width=True)
                                else:
                                    st.warning("Cannot generate comparison histogram: IRR range too narrow.")
                elif scenario_a == scenario_b:
                    st.warning("Please select two different scenarios to compare.")

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
    main()
