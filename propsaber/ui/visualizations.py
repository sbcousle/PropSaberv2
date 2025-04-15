# propsaber/ui/visualizations.py
"""
Contains functions for generating Plotly visualizations based on simulation results.
Extracted and adapted from the main script.
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
import logging
from typing import List, Dict, Optional, Tuple
from propsaber.core.utils import simulation_error_handler

logger = logging.getLogger(__name__)

# --- IRR Distribution Plot ---
@simulation_error_handler
def plot_irr_distribution(
    irr_values: List[float],
    title: str,
    mean_irr: float,
    median_irr: float,
    p05_irr: float,
    p95_irr: float,
    color: str = 'blue',
    bins: int = 30,
    x_range: Optional[Tuple[float, float]] = None,
    percent_format: bool = True
) -> go.Figure:
    """
    Generates a histogram Plotly figure for IRR distribution.

    Args:
        irr_values: List of finite IRR values.
        title: Title for the plot.
        mean_irr: Calculated mean IRR.
        median_irr: Calculated median IRR.
        p05_irr: Calculated 5th percentile IRR.
        p95_irr: Calculated 95th percentile IRR.
        color: Bar color for the histogram.
        bins: Number of bins for the histogram.
        x_range: Optional tuple (min, max) to set the x-axis range.
        percent_format: If True, format axis and annotations as percentages.

    Returns:
        A Plotly Figure object. Returns an empty figure if irr_values is empty.
    """
    if not irr_values:
        logger.warning(f"plot_irr_distribution: Empty irr_values for {title}")
        fig = go.Figure()
        fig.update_layout(title=f"{title} (No Data)", xaxis_title="IRR", yaxis_title="% of Instances")
        return fig

    try:
        # Calculate histogram data
        hist_data, bin_edges = np.histogram(irr_values, bins=bins, range=x_range) # Use range if provided
        bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
        total_count = sum(hist_data)
        percentages = (hist_data / total_count * 100.0) if total_count > 0 else np.zeros_like(hist_data)
        max_y = max(percentages) * 1.20 if any(p > 0 for p in percentages) else 1.0 # Dynamic y-axis scaling

        fig = go.Figure()
        # Add histogram bars
        fig.add_trace(go.Bar(x=bin_centers, y=percentages, marker_color=color, opacity=0.8, name="Frequency"))

        # --- Add Vertical Lines and Annotations for Stats ---
        annotation_format = "{:.1%}" if percent_format else "{:.2f}"
        annotation_y_offset = max_y * 0.08 # Adjust offset based on max bar height

        # Mean Line & Annotation
        if np.isfinite(mean_irr):
            fig.add_trace(go.Scatter(x=[mean_irr, mean_irr], y=[0, max_y*0.90], mode="lines",
                                     line=dict(color="red", dash="dash"), name="Mean"))
            fig.add_annotation(x=mean_irr, y=max_y, text=f"Mean: {annotation_format.format(mean_irr)}",
                               showarrow=False, font=dict(color="red"), yshift=annotation_y_offset, yanchor="bottom")

        # Median Line & Annotation
        if np.isfinite(median_irr):
            fig.add_trace(go.Scatter(x=[median_irr, median_irr], y=[0, max_y*0.90], mode="lines",
                                     line=dict(color="purple", dash="dash"), name="Median"))
            # Position median annotation lower to avoid overlap
            fig.add_annotation(x=median_irr, y=max_y, text=f"Median: {annotation_format.format(median_irr)}",
                               showarrow=False, font=dict(color="purple"), yshift=-annotation_y_offset*10.0, yanchor="bottom") # Adjust yshift

        # Percentile Lines & Annotations
        for val, label, y_shift_factor in zip([p05_irr, p95_irr], ["5th", "95th"], [1, 2]): # Stagger Pctl annotations
             if np.isfinite(val):
                 fig.add_trace(go.Scatter(x=[val, val], y=[0, max_y*0.90], mode="lines",
                                          line=dict(color="darkgrey", dash="dot"), name=f"{label} Pctl"))
                 fig.add_annotation(x=val, y=max_y, text=f"{label}: {annotation_format.format(val)}",
                                    showarrow=False, font=dict(color="darkgrey", size=10),
                                    yanchor="bottom", yshift=annotation_y_offset * y_shift_factor) # Use factor

        # --- Layout Updates ---
        fig.update_layout(
            title=title, xaxis_title="IRR", yaxis_title="% of Instances",
            bargap=0.1, showlegend=False, template="plotly_white", yaxis_ticksuffix="%"
        )

        # Apply x-axis formatting and range if specified
        x_axis_config = {}
        if percent_format:
            x_axis_config['tickformat'] = ".0%" # Format x-axis as percentage
        if x_range:
            if isinstance(x_range, (list, tuple)) and len(x_range) == 2:
                x_axis_config['range'] = x_range
            else:
                logger.warning(f"Invalid x_range provided for IRR plot: {x_range}")
        if x_axis_config:
            fig.update_xaxes(x_axis_config)

        return fig

    except Exception as e:
        logger.error(f"Error creating IRR distribution plot '{title}': {e}", exc_info=True)
        # Return an empty figure with error title on failure
        fig = go.Figure()
        fig.update_layout(title=f"{title} (Plotting Error)", xaxis_title="IRR", yaxis_title="% of Instances")
        return fig


# --- Rent vs Normal Plot ---
@simulation_error_handler
def plot_rent_vs_normal(
    years: List[int],
    market_p05: List[float],
    market_p50: List[float],
    market_p95: List[float],
    normal_p50: List[float]
) -> go.Figure:
    """
    Plots the distribution of Projected Market Rent vs the Fair Value Rent over time.

    Args:
        years: List of simulation years.
        market_p05: List of 5th percentile market rent values per year.
        market_p50: List of 50th percentile (median) market rent values per year.
        market_p95: List of 95th percentile market rent values per year.
        normal_p50: List of 50th percentile (median) fair value rent values per year.

    Returns:
        A Plotly Figure object.
    """
    fig = go.Figure()
    try:
        # Add 5th-95th percentile band for Projected Market Rent
        fig.add_trace(go.Scatter(
            x=years + years[::-1], y=market_p95 + market_p05[::-1], fill='toself',
            fillcolor='rgba(0,176,246,0.2)', line=dict(color='rgba(255,255,255,0)'),
            hoverinfo='skip', showlegend=True, name='Projected Market Rent (5th-95th Pctl)'
        ))
        # Add median line for Projected Market Rent
        fig.add_trace(go.Scatter(
            x=years, y=market_p50, line=dict(color='rgb(0,176,246)', width=2.5),
            mode='lines', name='Projected Market Rent (Median)'
        ))
        # Add median line for Fair Value Rent
        fig.add_trace(go.Scatter(
            x=years, y=normal_p50, line=dict(color='rgb(231,107,243)', width=2, dash='dash'),
            mode='lines', name='Fair Value Rent (Median)'
        ))

        # Update layout
        fig.update_layout(
            title="Projected Market Rent vs. Fair Value Rent",
            xaxis_title="Year", yaxis_title="Rent per Unit ($/Month)",
            template="plotly_white", hovermode="x unified",
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
            yaxis=dict(tickprefix='$', tickformat=',.0f') # Format y-axis as currency
        )
    except Exception as e:
        logger.error(f"Error creating rent vs normal plot: {e}", exc_info=True)
        # Return an empty figure with error title on failure
        fig = go.Figure()
        fig.update_layout(title="Rent vs Fair Value Plot (Plotting Error)", xaxis_title="Year", yaxis_title="Rent per Unit ($/Month)")

    return fig

# --- Vacancy Distribution Plot ---
@simulation_error_handler
def plot_vacancy_distribution(vacancy_plot_df: Optional[pd.DataFrame]) -> go.Figure:
    """
    Generates a box plot showing the distribution of vacancy rates per year.

    Args:
        vacancy_plot_df: Pandas DataFrame with columns 'Year' and 'Vacancy Rate',
                         containing vacancy rates for each simulation run and year.

    Returns:
        A Plotly Figure object. Returns an empty figure if data is missing or invalid.
    """
    if vacancy_plot_df is None or vacancy_plot_df.empty or not all(c in vacancy_plot_df.columns for c in ['Year', 'Vacancy Rate']):
        logger.warning("plot_vacancy_distribution: Invalid or empty DataFrame.")
        fig = go.Figure()
        fig.update_layout(title="Vacancy Rate Distribution (No Data)", xaxis_title="Year", yaxis_title="Vacancy Rate")
        return fig

    try:
        # Create box plot using Plotly Express
        fig = px.box(vacancy_plot_df, x="Year", y="Vacancy Rate", points="outliers",
                     color="Year", color_discrete_sequence=px.colors.sequential.Viridis) # Use a color sequence

        # Update layout
        fig.update_layout(
            title="Distribution of Simulated Vacancy Rate per Year",
            xaxis_title="Year", yaxis_title="Vacancy Rate",
            template="plotly_white", font=dict(size=12),
            showlegend=False, # No need for legend with color mapped to year
            yaxis=dict(tickformat=".1%") # Format y-axis as percentage
        )
        return fig
    except Exception as e:
        logger.error(f"Error creating vacancy distribution plot: {e}", exc_info=True)
        # Return an empty figure with error title on failure
        fig = go.Figure()
        fig.update_layout(title="Vacancy Rate Distribution (Plotting Error)", xaxis_title="Year", yaxis_title="Vacancy Rate")
        return fig

# --- Terminal Growth vs Exit Cap Plot ---
@simulation_error_handler
def plot_terminal_growth_vs_exit_cap(scatter_data: Dict[str, List[float]]) -> go.Figure:
    """
    Generates a scatter plot showing the relationship between terminal rent growth and exit cap rate.

    Args:
        scatter_data: Dictionary containing lists for 'term_rent_growth_pct' and 'exit_cap_rate_pct'.

    Returns:
        A Plotly Figure object. Returns an empty figure if data is missing or invalid.
    """
    required_keys = ["term_rent_growth_pct", "exit_cap_rate_pct"]
    if not scatter_data or not all(k in scatter_data for k in required_keys) or not scatter_data["term_rent_growth_pct"] or not scatter_data["exit_cap_rate_pct"]:
        logger.warning("Scatter plot data missing or empty.")
        fig = go.Figure()
        fig.update_layout(title="Terminal Rent Growth vs Exit Cap (No Data)", xaxis_title="Terminal Year Rent Growth (%)", yaxis_title="Simulated Exit Cap Rate (%)")
        return fig

    try:
        df_scatter = pd.DataFrame(scatter_data)
        if df_scatter.empty:
            logger.warning("Scatter plot DataFrame empty after creation.")
            fig = go.Figure()
            fig.update_layout(title="Terminal Rent Growth vs Exit Cap (No Data)", xaxis_title="Terminal Year Rent Growth (%)", yaxis_title="Simulated Exit Cap Rate (%)")
            return fig

        # Calculate correlation if enough data points exist
        corr_coef_str = ""
        if len(df_scatter) > 1:
            try:
                # Ensure columns exist before calculating correlation
                if "term_rent_growth_pct" in df_scatter and "exit_cap_rate_pct" in df_scatter:
                    corr_coef = df_scatter["term_rent_growth_pct"].corr(df_scatter["exit_cap_rate_pct"])
                    if np.isfinite(corr_coef):
                        corr_coef_str = f"(Corr: {corr_coef:.2f})"
                else:
                     logger.warning("Required columns missing for scatter plot correlation.")
            except Exception as e:
                logger.error(f"Scatter plot correlation calculation error: {e}")

        # Create scatter plot with trendline using Plotly Express
        fig = px.scatter(df_scatter, x="term_rent_growth_pct", y="exit_cap_rate_pct", opacity=0.6,
                         trendline="ols", trendline_color_override="red") # Add OLS trendline

        # Update layout
        fig.update_layout(
            title=f"Terminal Year Rent Growth vs Exit Cap Rate {corr_coef_str}",
            xaxis_title="Terminal Year Rent Growth (%)", yaxis_title="Simulated Exit Cap Rate (%)",
            template="plotly_white",
            xaxis_ticksuffix="%", yaxis_ticksuffix="%" # Add % suffix to axes
        )
        return fig

    except Exception as e:
        logger.error(f"Error creating terminal growth vs exit cap plot: {e}", exc_info=True)
        # Return an empty figure with error title on failure
        fig = go.Figure()
        fig.update_layout(title="Terminal Rent Growth vs Exit Cap (Plotting Error)", xaxis_title="Terminal Year Rent Growth (%)", yaxis_title="Simulated Exit Cap Rate (%)")
        return fig


# --- Simulated SOFR Distribution Plot ---
@simulation_error_handler
# In propsaber/ui/visualizations.py

@simulation_error_handler
def plot_simulated_sofr_distribution(
    years_list: List[int],
    underlying_sofr_paths: List[List[float]], # List of paths (each path is a list of rates for years_list)
    forward_rates_input: List[float], # List of forward curve rates corresponding to years_list
    refi_year: Optional[int] = None # <<< ADDED: Optional refinance year argument
) -> go.Figure:
    """
    Plots the distribution of the simulated underlying SOFR rate (before spread)
    against the input forward curve. Optionally adds a vertical line for the refi year.

    Args:
        years_list: List of simulation years.
        underlying_sofr_paths: List of lists, where each inner list is a simulated SOFR path.
        forward_rates_input: List of the input forward curve rates for the simulation years.
        refi_year: Optional integer indicating the year of refinance.

    Returns:
        A Plotly Figure object. Returns an empty figure if data is insufficient.
    """
    fig = go.Figure() # Initialize figure at the start

    # Basic data validation
    if not underlying_sofr_paths or not years_list or len(underlying_sofr_paths[0]) != len(years_list):
        logger.warning("Insufficient or mismatched data for plotting SOFR distribution.")
        fig.update_layout(title="Simulated SOFR Distribution (No Data)", xaxis_title="Year", yaxis_title="Simulated Base SOFR Rate")
        return fig

    try:
        # --- Data Preparation ---
        sofr_array = np.array(underlying_sofr_paths)
        p5 = np.percentile(sofr_array, 5, axis=0)
        median = np.percentile(sofr_array, 50, axis=0)
        p95 = np.percentile(sofr_array, 95, axis=0)

        # --- Add Traces ---
        # 5th-95th percentile band
        fig.add_trace(go.Scatter(
            x=years_list + years_list[::-1], y=list(p95) + list(p5[::-1]),
            fill='toself', fillcolor='rgba(0,100,80,0.2)', line=dict(color='rgba(255,255,255,0)'),
            hoverinfo='skip', showlegend=True, name='Simulated Base SOFR Rate (5th-95th Pctl)'
        ))
        # Median simulated line
        fig.add_trace(go.Scatter(
            x=years_list, y=median, line=dict(color='rgb(0,100,80)', width=2.5),
            mode='lines', name='Median Simulated Base SOFR Rate'
        ))
        # Input forward curve line
        if forward_rates_input and len(forward_rates_input) == len(years_list) and any(np.isfinite(fr) for fr in forward_rates_input):
            fig.add_trace(go.Scatter(
                x=years_list, y=forward_rates_input, mode='lines',
                line=dict(color='grey', dash='dot', width=2), name='Forward SOFR Curve (Input)'
            ))

        # --- <<< ADDED: Add Vertical Line for Refinance Year >>> ---
        if refi_year is not None and refi_year in years_list:
            fig.add_vline(
                x=refi_year,
                line_width=1.5,
                line_dash="dash",
                line_color="grey", # Using same style as the LTV plot marker
                annotation_text=f"Refi Yr {refi_year}",
                annotation_position="top right",
                annotation_font_size=10,
                annotation_font_color="grey"
            )
        # --- <<< END ADDED BLOCK >>> ---

        # --- Update Layout ---
        fig.update_layout(
            title="Simulated Underlying SOFR Rate Distribution (Excl. Spread)",
            xaxis_title="Year", yaxis_title="Underlying Simulated Base SOFR Rate",
            template="plotly_white", hovermode="x unified",
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
            yaxis_tickformat=".1%"
        )
        return fig

    except Exception as e:
        logger.error(f"Error creating simulated SOFR distribution plot: {e}", exc_info=True)
        fig.update_layout(title="Simulated SOFR Distribution (Plotting Error)", xaxis_title="Year", yaxis_title="Simulated Base SOFR Rate")
        return fig # Return the figure object even on error, but with error title

@simulation_error_handler
def plot_loan_balance_distribution(
    years: List[int],
    loan_balance_paths: List[List[float]],
    ltv_paths: List[List[float]],
    refi_year: Optional[int] = None # <<< ADDED: Optional refinance year argument
) -> go.Figure:
    """
    Plots the distribution of loan balances and LTV ratios over time.
    Optionally adds a vertical line indicating the refinance year.

    Args:
        years: List of simulation years.
        loan_balance_paths: List of lists, where each inner list is a loan balance path.
        ltv_paths: List of lists, where each inner list is an LTV ratio path.
        refi_year: Optional integer indicating the year of refinance.

    Returns:
        A Plotly Figure object. Returns an empty figure if data is insufficient.
    """
    fig = go.Figure() # Initialize figure at the start

    # Basic data validation
    if not years or not loan_balance_paths or not ltv_paths or len(loan_balance_paths[0]) != len(years):
        logger.warning("Insufficient or mismatched data for plotting loan balance/LTV.")
        fig.update_layout(
            title="Loan Balance and LTV Distribution (No Data)",
            xaxis_title="Year", yaxis_title="Loan Balance ($)"
        )
        return fig

    try:
        # --- Data Preparation ---
        loan_balance_array = np.array(loan_balance_paths)
        ltv_array = np.array(ltv_paths)
        balance_median = np.median(loan_balance_array, axis=0)
        balance_p5 = np.percentile(loan_balance_array, 5, axis=0)
        balance_p95 = np.percentile(loan_balance_array, 95, axis=0)
        ltv_median = np.median(ltv_array, axis=0)
        ltv_p5 = np.percentile(ltv_array, 5, axis=0)
        ltv_p95 = np.percentile(ltv_array, 95, axis=0)

        # --- Create Subplot with Secondary Y-axis ---
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        # --- Add Loan Balance Traces (Primary Y-axis) ---
        fig.add_trace(
            go.Scatter(
                x=years + years[::-1], y=list(balance_p95) + list(balance_p5[::-1]),
                fill='toself', fillcolor='rgba(0,100,80,0.2)', line=dict(color='rgba(255,255,255,0)'),
                hoverinfo='skip', name='Loan Balance (5th-95th Pctl)', showlegend=True
            ),
            secondary_y=False
        )
        fig.add_trace(
            go.Scatter(
                x=years, y=balance_median, line=dict(color='rgb(0,100,80)', width=2.5),
                name='Median Loan Balance', mode='lines'
            ),
            secondary_y=False
        )

        # --- Add LTV Traces (Secondary Y-axis) ---
        fig.add_trace(
            go.Scatter(
                x=years + years[::-1], y=list(ltv_p95) + list(ltv_p5[::-1]),
                fill='toself', fillcolor='rgba(255,165,0,0.2)', line=dict(color='rgba(255,255,255,0)'),
                hoverinfo='skip', name='LTV Ratio (5th-95th Pctl)', showlegend=True
            ),
            secondary_y=True
        )
        fig.add_trace(
            go.Scatter(
                x=years, y=ltv_median, line=dict(color='rgb(255,165,0)', width=2.5),
                name='Median LTV Ratio', mode='lines'
            ),
            secondary_y=True
        )

        # --- <<< ADDED: Add Vertical Line for Refinance Year >>> ---
        if refi_year is not None and refi_year in years:
            fig.add_vline(
                x=refi_year,
                line_width=1.5,
                line_dash="dash",
                line_color="grey",
                annotation_text=f"Refi Yr {refi_year}",
                annotation_position="top right", # Position annotation
                annotation_font_size=10,
                annotation_font_color="grey"
            )
        # --- <<< END ADDED BLOCK >>> ---

        # --- Update Layout ---
        fig.update_layout(
            title="Loan Balance and LTV Ratio Over Time",
            xaxis_title="Year", template="plotly_white", hovermode="x unified",
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
        )
        fig.update_yaxes(title_text="Loan Balance ($)", secondary_y=False, tickprefix="$", tickformat=",.0f")
        fig.update_yaxes(title_text="LTV Ratio", secondary_y=True, tickformat=".1%")

        return fig

    except Exception as e:
        logger.error(f"Error creating loan balance/LTV plot: {e}", exc_info=True)
        # Return the fig object initialized earlier, but update title to show error
        fig.update_layout(
            title="Loan Balance and LTV Distribution (Plotting Error)",
            xaxis_title="Year", yaxis_title="Loan Balance ($)"
        )
        return fig # Return the figure object even on error, but with error title
