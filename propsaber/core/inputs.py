# propsaber/core/inputs.py
"""
Define the SimulationInputs dataclass to hold all simulation parameters.
Includes calculated properties for convenience. Extracted from the main script.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, Any
import numpy as np
import logging

# Import necessary constants from constants.py
from .constants import LOAN_TYPE_IO, MONTHS_PER_YEAR, FLOAT_ATOL

logger = logging.getLogger(__name__)


@dataclass
class SimulationInputs:
    """Holds all input parameters for the simulation."""
    # --- Property & Setup ---
    purchase_price: float = 20_000_000.0
    num_units: int = 100
    hold_period: int = 10 # Max value will be increased in UI
    num_simulations: int = 1000 # Note: This is often controlled outside the core inputs

    # --- Rent & Market ---
    base_rent: float = 1800.0 # Initial Market Rent ($/Unit/Mo)
    market_rent_deviation_pct: float = -0.05 # Initial Market Rent Premium/Discount to Fair Value (%)
    market_convergence_years: int = 3 # Years to Normalize Rent
    normal_growth_mean: float = 2.5 # Avg Fair Value Rent Growth (Normal, %/Yr)
    normal_growth_vol: float = 1.5 # Fair Value Rent Growth Volatility (Normal, % pts)
    recession_growth_mean: float = -2.0 # Avg Fair Value Rent Growth (Recession, %/Yr)
    recession_growth_vol: float = 5.0 # Fair Value Rent Growth Volatility (Recession, % pts)
    transition_normal_to_recession: float = 0.00 # Prob. Normal → Recession (%/Yr)
    transition_recession_to_normal: float = 0.60 # Prob. Recession → Normal (%/Yr)

    # --- Vacancy ---
    current_vacancy: float = 0.10 # Starting Vacancy Rate (%)
    stabilized_vacancy: float = 0.05 # Target Long-Term Vacancy Rate (%)
    vacancy_reversion_speed: float = 0.30 # Vacancy Reversion Speed (unitless)
    vacancy_volatility: float = 0.015 # Vacancy Rate Volatility (% pts/Yr, absolute std dev for OU process)

    # --- Other Income ---
    mean_other_income: float = 56_000.0 # Initial Annual Other Income ($)
    mean_other_income_growth: float = 3.0 # Avg Other Income Growth (%/Yr)
    other_income_stddev: float = 1.5 # Other Income Growth Volatility (% pts)

    # --- Operating Expenses ---
    mean_expense: float = 1_000_000.0 # Initial Annual Operating Expenses ($)
    mean_expense_growth: float = 3.0 # Average OpEx Growth (%/Yr)
    expense_stddev: float = 1.5 # OpEx Growth Volatility (% pts)

    # --- Capital Expenditures ---
    capex_per_unit_yr: float = 250.0 # Initial Annual CapEx Reserve ($/Unit/Yr)
    mean_capex_growth: float = 3.0 # Average CapEx Growth (%/Yr)
    capex_stddev: float = 5.0 # CapEx Growth Volatility (% pts)

    # --- Simulation Dynamics ---
    growth_persistence_phi: float = 0.5 # Cyclical Persistence (Growth) - AR(1) factor for GBM shocks
    use_correlations: bool = True # Use Correlations Between Shocks

    # --- Correlations (only used if use_correlations is True) ---
    corr_rent_expense: float = 0.3 # Corr: Rent & Expense Shocks
    corr_rent_other_income: float = 0.4 # Corr: Rent & Other Income Shocks
    corr_rent_vacancy: float = -0.5 # Corr: Rent & Vacancy Shocks

    # --- Financing ---
    loan_to_cost: float = 0.70 # Loan-to-Cost Ratio (%)
    initial_loan_costs_pct: float = 0.01 # Initial Loan Costs (% of Loan Amount) - NEW
    is_variable_rate: bool = False # Rate Type (False=Fixed, True=Floating)
    # Fixed Rate Specific (used if is_variable_rate is False)
    interest_rate: float = 0.06 # Fixed Loan Interest Rate (%/Yr)
    loan_type: str = LOAN_TYPE_IO # Loan Type (Interest Only or Amortizing) - Fixed Rate Only
    loan_term_yrs: int = 30 # Amortization Period (Years) - Fixed Rate, Amortizing Only
    # Floating Rate Specific (used if is_variable_rate is True)
    sofr_spread: float = 0.02 # Spread Over SOFR (%/Yr)
    volatility_scalar: float = 1.0 # Floating Rate Volatility Factor (scales derived volatility)
    sofr_floor: float = 0.0 # SOFR Floor (%/Yr, applied before spread)
    rate_persistence_phi: float = 0.5 # Cyclical Persistence (Rates) - AR(1) factor for rate simulation

    # --- Refinancing (NEW SECTION) ---
    enable_refinancing: bool = False # Toggle for refinancing logic
    refi_year: int = 10 # Year refinance occurs (if enabled)
    refi_fixed_rate_spread_to_sofr: float = 2.5 # Spread over simulated SOFR for new fixed rate (%)
    refi_new_ltv: float = 0.70 # Target LTV for the new loan (%)
    refi_new_amort_period: int = 30 # Amortization period for new loan (Years)
    refi_costs_pct_loan: float = 0.01 # Refi costs as % of new loan amount

    # --- Exit Assumptions ---
    mean_exit_cap_rate: float = 5.5 # Average Exit Cap Rate (%)
    exit_cap_rate_stddev: float = 0.5 # Exit Cap Rate Volatility (% pts)
    transaction_cost_pct: float = 0.010 # Transaction Cost on Sale (%)
    exit_cap_rent_growth_sensitivity: float = -0.10 # Exit Cap Adj. for Rent Growth (unitless sensitivity)
    exit_floor_value: float = 2_000_000.0 # Gross Exit Value Floor ($)

    # --- Risk Metrics ---
    risk_free_rate: float = 0.04 # Risk-Free Rate (%/Yr) for Sharpe Ratio
    hurdle_rate: float = 0.08 # Hurdle Rate (% IRR) for Prob. Below Hurdle

    # --- Calculated Properties ---
    @property
    def initial_noi(self) -> float:
        """Calculates the estimated Net Operating Income at Year 0 based on inputs."""
        try:
            potential_gross_rent = self.num_units * self.base_rent * MONTHS_PER_YEAR
            effective_gross_rent = potential_gross_rent * (1 - self.current_vacancy)
            effective_gross_income = effective_gross_rent + self.mean_other_income
            noi = effective_gross_income - self.mean_expense
            return noi if np.isfinite(noi) else 0.0 # Return 0 if calculation results in NaN/Inf
        except Exception as e:
            logger.error(f"Error calculating initial_noi: {e}", exc_info=True)
            return 0.0 # Return 0 on error

    @property
    def initial_capex(self) -> float:
        """Calculates the estimated initial annual Capital Expenditures based on inputs."""
        try:
            capex = self.num_units * self.capex_per_unit_yr
            return capex if np.isfinite(capex) else 0.0
        except Exception as e:
            logger.error(f"Error calculating initial_capex: {e}", exc_info=True)
            return 0.0

    @property
    def loan_amount(self) -> float:
        """Calculates the initial loan amount based on purchase price and LTC."""
        try:
            # Ensure purchase price is positive to avoid division by zero or negative loans
            if self.purchase_price > FLOAT_ATOL:
                loan = self.purchase_price * self.loan_to_cost
                return loan if np.isfinite(loan) else 0.0
            else:
                return 0.0
        except Exception as e:
            logger.error(f"Error calculating loan_amount: {e}", exc_info=True)
            return 0.0

    @property
    def initial_loan_costs(self) -> float:
        """Calculates the initial loan closing costs."""
        try:
            costs = self.loan_amount * self.initial_loan_costs_pct
            return costs if np.isfinite(costs) else 0.0
        except Exception as e:
            logger.error(f"Error calculating initial_loan_costs: {e}", exc_info=True)
            return 0.0

    @property
    def initial_equity(self) -> float:
        """Calculates the initial equity required, including loan costs."""
        try:
            equity = self.purchase_price - self.loan_amount + self.initial_loan_costs
            return equity if np.isfinite(equity) else 0.0
        except Exception as e:
            logger.error(f"Error calculating initial_equity: {e}", exc_info=True)
            return 0.0


    def to_dict(self) -> Dict[str, Any]:
        """Converts the dataclass instance to a dictionary."""
        # Use asdict for proper dataclass conversion
        return asdict(self)

