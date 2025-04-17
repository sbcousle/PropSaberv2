"""
Contains functions for generating Plotly visualizations based on simulation results.
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
        hist_data, bin_edges = np.histogram(irr_values, bins=bins, range=x_range)
        bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
        total_count = sum(hist_data)
        percentages = (hist_data / total_count * 100.0) if total_count > 0 else np.zeros_like(hist_data)
        max_y = max(percentages) * 1.20 if any(p > 0 for p in percentages) else 1.0

        fig = go.Figure()
        fig.add_trace(go.Bar(x=bin_centers, y=percentages, marker_color=color, opacity=0.8, name="Frequency"))

        annotation_format = "{:.1%}" if percent_format else "{:.2f}"
        annotation_y_offset = max_y * 0.08

        # Mean Line & Annotation
        if np.isfinite(mean_irr):
            fig.add_shape(
                type="line",
                x0=mean_irr, x1=mean_irr,
                y0=0, y1=max_y * 0.90,
                line=dict(color="red", dash="dash", width=1.5)
            )
            fig.add_annotation(
                x=mean_irr,
                y=max_y * 1.05,
                text=f"Mean: {annotation_format.format(mean_irr)}",
                showarrow=False,
                font=dict(color="red"),
                yshift=annotation_y_offset * 2,
                yanchor="bottom"
            )

        # Median Line & Annotation
        if np.isfinite(median_irr):
            fig.add_shape(
                type="line",
                x0=median_irr, x1=median_irr,
                y0=0, y1=max_y * 0.90,
                line=dict(color="purple", dash="dash", width=1.5)
            )
            fig.add_annotation(
                x=median_irr,
                y=max_y * 1.05,
                text=f"Median: {annotation_format.format(median_irr)}",
                showarrow=False,
                font=dict(color="purple"),
                yshift=-annotation_y_offset * 12,
                yanchor="bottom"
            )

        # Percentile Lines & Annotations
        for val, label, y_shift_factor in zip([p05_irr, p95_irr], ["5th", "95th"], [1.5, 3]):
            if np.isfinite(val):
                fig.add_shape(
                    type="line",
                    x0=val, x1=val,
                    y0=0, y1=max_y * 0.90,
                    line=dict(color="darkgrey", dash="dot", width=1.5)
                )
                fig.add_annotation(
                    x=val,
                    y=max_y * 1.05,
                    text=f"{label}: {annotation_format.format(val)}",
                    showarrow=False,
                    font=dict(color="darkgrey", size=10),
                    yshift=annotation_y_offset * y_shift_factor,
                    yanchor="bottom"
                )

        fig.update_layout(
            title=title,
            xaxis_title="IRR",
            yaxis_title="% of Instances",
            bargap=0.1,
            showlegend=False,
            template="plotly_white",
            yaxis_ticksuffix="%"
        )

        x_axis_config = {}
        if percent_format:
            x_axis_config['tickformat'] = ".0%"
        if x_range and isinstance(x_range, (list, tuple)) and len(x_range) == 2:
            x_axis_config['range'] = x_range
        if x_axis_config:
            fig.update_xaxes(**x_axis_config)

        return fig

    except Exception as e:
        logger.error(f"Error creating IRR distribution plot '{title}': {e}", exc_info=True)
        fig = go.Figure()
        fig.update_layout(title=f"{title} (Plotting Error)", xaxis_title="IRR", yaxis_title="% of Instances")
        return fig

# --- Multiple Distribution Plot ---
@simulation_error_handler
def plot_multiple_distribution(
    multiple_values: List[float],
    title: str,
    mean_multiple: Optional[float] = None,
    median_multiple: Optional[float] = None,
    p05_multiple: Optional[float] = None,
    p95_multiple: Optional[float] = None,
    color: str = 'skyblue',
    bins: int = 30,
    x_range: Optional[Tuple[float, float]] = None
) -> go.Figure:
    """
    Generates a histogram Plotly figure for equity multiple distributions.

    Args:
        multiple_values: List of finite multiple values.
        title: Title for the chart.
        mean_multiple: Calculated mean multiple.
        median_multiple: Calculated median multiple.
        p05_multiple: Calculated 5th percentile multiple.
        p95_multiple: Calculated 95th percentile multiple.
        color: Color for the histogram bars.
        bins: Number of bins for the histogram.
        x_range: Optional tuple (min, max) to set the x-axis range.

    Returns:
        A Plotly Figure object. Returns an empty figure if multiple_values is empty.
    """
    if not multiple_values:
        logger.warning(f"plot_multiple_distribution: Empty multiple_values for {title}")
        fig = go.Figure()
        fig.update_layout(title=f"{title} (No Data)", xaxis_title="Equity Multiple", yaxis_title="% of Instances")
        return fig

    try:
        hist_data, bin_edges = np.histogram(multiple_values, bins=bins, range=x_range)
        bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
        total_count = sum(hist_data)
        percentages = (hist_data / total_count * 100.0) if total_count > 0 else np.zeros_like(hist_data)
        max_y = max(percentages) * 1.20 if any(p > 0 for p in percentages) else 1.0

        fig = go.Figure()
        fig.add_trace(go.Bar(x=bin_centers, y=percentages, marker_color=color, opacity=0.8, name="Frequency"))

        annotation_y_offset = max_y * 0.08

        # Mean Line & Annotation
        if mean_multiple is not None and np.isfinite(mean_multiple):
            fig.add_shape(
                type="line",
                x0=mean_multiple, x1=mean_multiple,
                y0=0, y1=max_y * 0.90,
                line=dict(color="red", dash="dash", width=1.5)
            )
            fig.add_annotation(
                x=mean_multiple,
                y=max_y * 1.05,
                text=f"Mean: {mean_multiple:.1f}x",
                showarrow=False,
                font=dict(color="red"),
                yshift=annotation_y_offset * 2,
                yanchor="bottom"
            )

        # Median Line & Annotation
        if median_multiple is not None and np.isfinite(median_multiple):
            fig.add_shape(
                type="line",
                x0=median_multiple, x1=median_multiple,
                y0=0, y1=max_y * 0.90,
                line=dict(color="purple", dash="dash", width=1.5)
            )
            fig.add_annotation(
                x=median_multiple,
                y=max_y * 1.05,
                text=f"Median: {median_multiple:.1f}x",
                showarrow=False,
                font=dict(color="purple"),
                yshift=-annotation_y_offset * 12,
                yanchor="bottom"
            )

        # Percentile Lines & Annotations
        for val, label, y_shift_factor in zip(
            [p05_multiple, p95_multiple], ["5th", "95th"], [1.5, 3]
        ):
            if val is not None and np.isfinite(val):
                fig.add_shape(
                    type="line",
                    x0=val, x1=val,
                    y0=0, y1=max_y * 0.90,
                    line=dict(color="darkgrey", dash="dot", width=1.5)
                )
                fig.add_annotation(
                    x=val,
                    y=max_y * 1.05,
                    text=f"{label}: {val:.1f}x",
                    showarrow=False,
                    font=dict(color="darkgrey", size=10),
                    yshift=annotation_y_offset * y_shift_factor,
                    yanchor="bottom"
                )

        fig.update_layout(
            title=title,
            xaxis_title="Equity Multiple",
            yaxis_title="% of Instances",
            bargap=0.1,
            showlegend=False,
            template="plotly_white",
            yaxis_ticksuffix="%",
            xaxis_tickformat=".1f",
            xaxis_ticksuffix="x"
        )

        if x_range and isinstance(x_range, (list, tuple)) and len(x_range) == 2:
            fig.update_xaxes(range=x_range)

        return fig

    except Exception as e:
        logger.error(f"Error creating multiple distribution plot '{title}': {e}", exc_info=True)
        fig = go.Figure()
        fig.update_layout(title=f"{title} (Plotting Error)", xaxis_title="Equity Multiple", yaxis_title="% of Instances")
        return fig

# --- Exit Distribution Plot ---
@simulation_error_handler
def plot_exit_distribution(
    values: List[float],
    title: str,
    mean_value: Optional[float] = None,
    median_value: Optional[float] = None,
    p05_value: Optional[float] = None,
    p95_value: Optional[float] = None,
    color: str = 'mediumseagreen',
    bins: int = 30,
    x_range: Optional[Tuple[float, float]] = None,
    is_currency: bool = False,
    is_percent: bool = False
) -> go.Figure:
    """
    Generates a histogram Plotly figure for exit value or cap rate distributions.

    Args:
        values: List of finite values (e.g., exit values or cap rates).
        title: Title for the chart.
        mean_value: Calculated mean value.
        median_value: Calculated median value.
        p05_value: Calculated 5th percentile value.
        p95_value: Calculated 95th percentile value.
        color: Color for the histogram bars.
        bins: Number of bins for the histogram.
        x_range: Optional tuple (min, max) to set the x-axis range.
        is_currency: If True, format as currency (e.g., for exit values).
        is_percent: If True, format as percentage (e.g., for cap rates).

    Returns:
        A Plotly Figure object. Returns an empty figure if values is invalid.
    """
    if not values or not all(np.isfinite(v) for v in values) or len(values) < 2:
        logger.warning(f"plot_exit_distribution: Empty or invalid values for {title}")
        fig = go.Figure()
        fig.update_layout(title=f"{title} (No Data)", xaxis_title="Value", yaxis_title="% of Instances")
        return fig

    try:
        # Log data range for debugging
        min_val, max_val = min(values), max(values)
        logger.info(f"plot_exit_distribution: {title}, min={min_val}, max={max_val}, count={len(values)}")

        # Handle identical or narrow range values
        unique_values = sorted(set(values))
        if len(unique_values) == 1:
            logger.warning(f"plot_exit_distribution: All values identical for {title}")
            fig = go.Figure()
            fig.add_trace(go.Bar(x=[values[0]], y=[100.0], marker_color=color, opacity=0.8, name="Frequency"))
            percentages = [100.0]
            max_y = 120.0
            bin_centers = [values[0]]
        else:
            # Set dynamic x_range for large values
            if not x_range:
                spread = max_val - min_val
                if spread < 1e-6:
                    spread = max(1e-6, max_val * 0.1)
                x_range = (min_val - spread * 0.1, max_val + spread * 0.1)

            # Normalize values for large numbers (if currency)
            if is_currency and max_val > 1e6:
                scaled_values = [v / 1e6 for v in values]  # Scale to millions
                scaled_x_range = (x_range[0] / 1e6, x_range[1] / 1e6) if x_range else None
                scaled_mean = mean_value / 1e6 if mean_value is not None and np.isfinite(mean_value) else None
                scaled_median = median_value / 1e6 if median_value is not None and np.isfinite(median_value) else None
                scaled_p05 = p05_value / 1e6 if p05_value is not None and np.isfinite(p05_value) else None
                scaled_p95 = p95_value / 1e6 if p95_value is not None and np.isfinite(p95_value) else None
                annotation_format = "${:,.0f}M"
                xaxis_title = "Net Exit Value ($M)"
                xaxis_config = {'tickformat': "$,.0f", 'ticksuffix': "M"}
            else:
                scaled_values = values
                scaled_x_range = x_range
                scaled_mean = mean_value
                scaled_median = median_value
                scaled_p05 = p05_value
                scaled_p95 = p95_value
                if is_currency:
                    annotation_format = "${:,.0f}"
                    xaxis_title = "Net Exit Value ($)"
                    xaxis_config = {'tickformat': "$,.0s"}
                elif is_percent:
                    annotation_format = "{:.2f}%"
                    xaxis_title = "Exit Cap Rate (%)"
                    xaxis_config = {'ticksuffix': "%", 'tickformat': ".2f"}
                else:
                    annotation_format = "{:.2f}"
                    xaxis_title = "Value"
                    xaxis_config = {'tickformat': ".2f"}

            hist_data, bin_edges = np.histogram(scaled_values, bins=bins, range=scaled_x_range)
            bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
            total_count = sum(hist_data)
            percentages = (hist_data / total_count * 100.0) if total_count > 0 else np.zeros_like(hist_data)
            max_y = max(percentages) * 1.20 if any(p > 0 for p in percentages) else 1.0
            fig = go.Figure()
            fig.add_trace(go.Bar(x=bin_centers, y=percentages, marker_color=color, opacity=0.8, name="Frequency"))

            annotation_y_offset = max_y * 0.08

            # Mean Line & Annotation
            if scaled_mean is not None and np.isfinite(scaled_mean):
                fig.add_trace(go.Scatter(
                    x=[scaled_mean, scaled_mean], y=[0, max_y*0.90],
                    mode="lines",
                    line=dict(color="red", dash="dash"),
                    name="Mean",
                    showlegend=False
                ))
                fig.add_annotation(
                    x=scaled_mean,
                    y=max_y,
                    text=f"Mean: {annotation_format.format(scaled_mean)}",
                    showarrow=False,
                    font=dict(color="red"),
                    yshift=annotation_y_offset,
                    yanchor="bottom"
                )

            # Median Line & Annotation
            if scaled_median is not None and np.isfinite(scaled_median):
                fig.add_trace(go.Scatter(
                    x=[scaled_median, scaled_median], y=[0, max_y*0.90],
                    mode="lines",
                    line=dict(color="purple", dash="dash"),
                    name="Median",
                    showlegend=False
                ))
                fig.add_annotation(
                    x=scaled_median,
                    y=max_y,
                    text=f"Median: {annotation_format.format(scaled_median)}",
                    showarrow=False,
                    font=dict(color="purple"),
                    yshift=-annotation_y_offset*10.0,
                    yanchor="bottom"
                )

            # Percentile Lines & Annotations
            for val, label, y_shift_factor in zip(
                [scaled_p05, scaled_p95], ["5th", "95th"], [1, 2]
            ):
                if val is not None and np.isfinite(val):
                    fig.add_trace(go.Scatter(
                        x=[val, val], y=[0, max_y*0.90],
                        mode="lines",
                        line=dict(color="darkgrey", dash="dot"),
                        name=f"{label} Pctl",
                        showlegend=False
                    ))
                    fig.add_annotation(
                        x=val,
                        y=max_y,
                        text=f"{label}: {annotation_format.format(val)}",
                        showarrow=False,
                        font=dict(color="darkgrey", size=10),
                        yshift=annotation_y_offset * y_shift_factor,
                        yanchor="bottom"
                    )

            fig.update_layout(
                title=title,
                xaxis_title=xaxis_title,
                yaxis_title="% of Instances",
                bargap=0.1,
                showlegend=False,
                template="plotly_white",
                yaxis_ticksuffix="%"
            )

            fig.update_xaxes(**xaxis_config)
            if scaled_x_range and isinstance(scaled_x_range, (list, tuple)) and len(scaled_x_range) == 2:
                fig.update_xaxes(range=scaled_x_range)

            return fig

    except Exception as e:
        logger.error(f"Error creating exit distribution plot '{title}': {e}, min={min_val}, max={max_val}, bins={bins}", exc_info=True)
        fig = go.Figure()
        fig.update_layout(title=f"{title} (Plotting Error)", xaxis_title=xaxis_title, yaxis_title="% of Instances")
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
        fig.add_trace(go.Scatter(
            x=years + years[::-1],
            y=market_p95 + market_p05[::-1],
            fill='toself',
            fillcolor='rgba(0,176,246,0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo='skip',
            showlegend=True,
            name='Projected Market Rent (5th-95th Pctl)'
        ))
        fig.add_trace(go.Scatter(
            x=years,
            y=market_p50,
            line=dict(color='rgb(0,176,246)', width=2.5),
            mode='lines',
            name='Projected Market Rent (Median)'
        ))
        fig.add_trace(go.Scatter(
            x=years,
            y=normal_p50,
            line=dict(color='rgb(231,107,243)', width=2, dash='dash'),
            mode='lines',
            name='Fair Value Rent (Median)'
        ))

        fig.update_layout(
            title="Projected Market Rent vs. Fair Value Rent",
            xaxis_title="Year",
            yaxis_title="Rent per Unit ($/Month)",
            template="plotly_white",
            hovermode="x unified",
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
            yaxis=dict(tickprefix='$', tickformat=',.0f')
        )
    except Exception as e:
        logger.error(f"Error creating rent vs normal plot: {e}", exc_info=True)
        fig.update_layout(
            title="Rent vs Fair Value Plot (Plotting Error)",
            xaxis_title="Year",
            yaxis_title="Rent per Unit ($/Month)"
        )

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
        fig.update_layout(
            title="Vacancy Rate Distribution (No Data)",
            xaxis_title="Year",
            yaxis_title="Vacancy Rate"
        )
        return fig

    try:
        fig = px.box(
            vacancy_plot_df,
            x="Year",
            y="Vacancy Rate",
            points="outliers",
            color="Year",
            color_discrete_sequence=px.colors.sequential.Viridis
        )

        fig.update_layout(
            title="Distribution of Simulated Vacancy Rate per Year",
            xaxis_title="Year",
            yaxis_title="Vacancy Rate",
            template="plotly_white",
            font=dict(size=12),
            showlegend=False,
            yaxis=dict(tickformat=".1%")
        )
        return fig
    except Exception as e:
        logger.error(f"Error creating vacancy distribution plot: {e}", exc_info=True)
        fig = go.Figure()
        fig.update_layout(
            title="Vacancy Rate Distribution (Plotting Error)",
            xaxis_title="Year",
            yaxis_title="Vacancy Rate"
        )
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
        fig.update_layout(
            title="Terminal Rent Growth vs Exit Cap (No Data)",
            xaxis_title="Terminal Year Rent Growth (%)",
            yaxis_title="Simulated Exit Cap Rate (%)"
        )
        return fig

    try:
        df_scatter = pd.DataFrame(scatter_data)
        if df_scatter.empty:
            logger.warning("Scatter plot DataFrame empty after creation.")
            fig = go.Figure()
            fig.update_layout(
                title="Terminal Rent Growth vs Exit Cap (No Data)",
                xaxis_title="Terminal Year Rent Growth (%)",
                yaxis_title="Simulated Exit Cap Rate (%)"
            )
            return fig

        corr_coef_str = ""
        if len(df_scatter) > 1 and "term_rent_growth_pct" in df_scatter and "exit_cap_rate_pct" in df_scatter:
            corr_coef = df_scatter["term_rent_growth_pct"].corr(df_scatter["exit_cap_rate_pct"])
            if np.isfinite(corr_coef):
                corr_coef_str = f"(Corr: {corr_coef:.2f})"

        fig = px.scatter(
            df_scatter,
            x="term_rent_growth_pct",
            y="exit_cap_rate_pct",
            opacity=0.6,
            trendline="ols",
            trendline_color_override="red"
        )

        fig.update_layout(
            title=f"Terminal Year Rent Growth vs Exit Cap Rate {corr_coef_str}",
            xaxis_title="Terminal Year Rent Growth (%)",
            yaxis_title="Simulated Exit Cap Rate (%)",
            template="plotly_white",
            xaxis_ticksuffix="%",
            yaxis_ticksuffix="%"
        )
        return fig

    except Exception as e:
        logger.error(f"Error creating terminal growth vs exit cap plot: {e}", exc_info=True)
        fig = go.Figure()
        fig.update_layout(
            title="Terminal Rent Growth vs Exit Cap (Plotting Error)",
            xaxis_title="Terminal Year Rent Growth (%)",
            yaxis_title="Simulated Exit Cap Rate (%)"
        )
        return fig

# --- Simulated SOFR Distribution Plot ---
@simulation_error_handler
def plot_simulated_sofr_distribution(
    years_list: List[int],
    underlying_sofr_paths: List[List[float]],
    forward_rates_input: List[float],
    refi_year: Optional[int] = None
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
    fig = go.Figure()

    if not underlying_sofr_paths or not years_list or len(underlying_sofr_paths[0]) != len(years_list):
        logger.warning("Insufficient or mismatched data for plotting SOFR distribution.")
        fig.update_layout(
            title="Simulated SOFR Distribution (No Data)",
            xaxis_title="Year",
            yaxis_title="Simulated Base SOFR Rate"
        )
        return fig

    try:
        sofr_array = np.array(underlying_sofr_paths)
        p5 = np.percentile(sofr_array, 5, axis=0)
        median = np.percentile(sofr_array, 50, axis=0)
        p95 = np.percentile(sofr_array, 95, axis=0)

        fig.add_trace(go.Scatter(
            x=years_list + years_list[::-1],
            y=list(p95) + list(p5[::-1]),
            fill='toself',
            fillcolor='rgba(0,100,80,0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo='skip',
            showlegend=True,
            name='Simulated Base SOFR Rate (5th-95th Pctl)'
        ))
        fig.add_trace(go.Scatter(
            x=years_list,
            y=median,
            line=dict(color='rgb(0,100,80)', width=2.5),
            mode='lines',
            name='Median Simulated Base SOFR Rate'
        ))
        if forward_rates_input and len(forward_rates_input) == len(years_list) and any(np.isfinite(fr) for fr in forward_rates_input):
            fig.add_trace(go.Scatter(
                x=years_list,
                y=forward_rates_input,
                mode='lines',
                line=dict(color='grey', dash='dot', width=2),
                name='Forward SOFR Curve (Input)'
            ))

        if refi_year is not None and refi_year in years_list:
            fig.add_vline(
                x=refi_year,
                line_width=1.5,
                line_dash="dash",
                line_color="grey",
                annotation_text=f"Refi Yr {refi_year}",
                annotation_position="top right",
                annotation_font_size=10,
                annotation_font_color="grey"
            )

        fig.update_layout(
            title="Simulated Underlying SOFR Rate Distribution (Excl. Spread)",
            xaxis_title="Year",
            yaxis_title="Underlying Simulated Base SOFR Rate",
            template="plotly_white",
            hovermode="x unified",
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
            yaxis_tickformat=".1%"
        )
        return fig

    except Exception as e:
        logger.error(f"Error creating simulated SOFR distribution plot: {e}", exc_info=True)
        fig.update_layout(
            title="Simulated SOFR Distribution (Plotting Error)",
            xaxis_title="Year",
            yaxis_title="Simulated Base SOFR Rate"
        )
        return fig

# --- Loan Balance Distribution Plot ---
@simulation_error_handler
def plot_loan_balance_distribution(
    years: List[int],
    loan_balance_paths: List[List[float]],
    ltv_paths: List[List[float]],
    refi_year: Optional[int] = None
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
    fig = go.Figure()

    if not years or not loan_balance_paths or not ltv_paths or len(loan_balance_paths[0]) != len(years):
        logger.warning("Insufficient or mismatched data for plotting loan balance/LTV.")
        fig.update_layout(
            title="Loan Balance and LTV Distribution (No Data)",
            xaxis_title="Year",
            yaxis_title="Loan Balance ($)"
        )
        return fig

    try:
        loan_balance_array = np.array(loan_balance_paths)
        ltv_array = np.array(ltv_paths)
        balance_median = np.median(loan_balance_array, axis=0)
        balance_p5 = np.percentile(loan_balance_array, 5, axis=0)
        balance_p95 = np.percentile(loan_balance_array, 95, axis=0)
        ltv_median = np.median(ltv_array, axis=0)
        ltv_p5 = np.percentile(ltv_array, 5, axis=0)
        ltv_p95 = np.percentile(ltv_array, 95, axis=0)

        fig = make_subplots(specs=[[{"secondary_y": True}]])

        fig.add_trace(
            go.Scatter(
                x=years + years[::-1],
                y=list(balance_p95) + list(balance_p5[::-1]),
                fill='toself',
                fillcolor='rgba(0,100,80,0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                hoverinfo='skip',
                name='Loan Balance (5th-95th Pctl)',
                showlegend=True
            ),
            secondary_y=False
        )
        fig.add_trace(
            go.Scatter(
                x=years,
                y=balance_median,
                line=dict(color='rgb(0,100,80)', width=2.5),
                name='Median Loan Balance',
                mode='lines'
            ),
            secondary_y=False
        )

        fig.add_trace(
            go.Scatter(
                x=years + years[::-1],
                y=list(ltv_p95) + list(ltv_p5[::-1]),
                fill='toself',
                fillcolor='rgba(255,165,0,0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                hoverinfo='skip',
                name='LTV Ratio (5th-95th Pctl)',
                showlegend=True
            ),
            secondary_y=True
        )
        fig.add_trace(
            go.Scatter(
                x=years,
                y=ltv_median,
                line=dict(color='rgb(255,165,0)', width=2.5),
                name='Median LTV Ratio',
                mode='lines'
            ),
            secondary_y=True
        )

        if refi_year is not None and refi_year in years:
            fig.add_vline(
                x=refi_year,
                line_width=1.5,
                line_dash="dash",
                line_color="grey",
                annotation_text=f"Refi Yr {refi_year}",
                annotation_position="top right",
                annotation_font_size=10,
                annotation_font_color="grey"
            )

        fig.update_layout(
            title="Loan Balance and LTV Ratio Over Time",
            xaxis_title="Year",
            template="plotly_white",
            hovermode="x unified",
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
        )
        fig.update_yaxes(
            title_text="Loan Balance ($)",
            secondary_y=False,
            tickprefix="$",
            tickformat=",.0f"
        )
        fig.update_yaxes(
            title_text="LTV Ratio",
            secondary_y=True,
            tickformat=".1%"
        )

        return fig

    except Exception as e:
        logger.error(f"Error creating loan balance/LTV plot: {e}", exc_info=True)
        fig.update_layout(
            title="Loan Balance and LTV Distribution (Plotting Error)",
            xaxis_title="Year",
            yaxis_title="Loan Balance ($)"
        )
        return fig
