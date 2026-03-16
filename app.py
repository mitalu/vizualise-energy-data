
import streamlit as st
import pandas as pd
import altair as alt
import numpy as np
from io import StringIO

# ---------------------------
# Page setup
# ---------------------------
st.set_page_config(page_title="Energy – Bars", layout="wide")
st.title("🔌 Energy Consumption – Bar Graphs by Day / Month / Year / Years")

st.markdown(
    """
This app expects **cumulative meter readings** (e.g., Wh or kWh) with a timestamp.
Consumption is computed as: **(period's last reading) − (previous period's last reading)**.
Use the sidebar to upload data, set columns, units, and choose the aggregation level.
"""
)

# ---------------------------
# Sidebar: inputs & options
# ---------------------------
with st.sidebar:
    st.header("⚙️ Data & Settings")

    uploaded = st.file_uploader(
        "Upload CSV", type=["csv"],
        help="CSV with at least a timestamp and a cumulative meter column."
    )
    time_col = st.text_input("Timestamp column", value="timestamp")

    st.caption("If your timestamps are UTC *and* you want local calendar boundaries, set a timezone.")
    tz = st.text_input("Timezone (optional)", value="", placeholder="e.g., Europe/Prague")

    st.divider()
    unit_divisor = st.number_input(
        "Unit divisor", min_value=0.000001, value=1000.0, step=0.1,
        help="Divide values by this. Example: Wh→kWh use 1000."
    )
    unit_label = st.text_input("Unit label", value="kWh", help="E.g., kWh")

    drop_neg = st.checkbox("Drop negative diffs (meter resets)", value=True)

    st.divider()
    st.caption("Optional: paste CSV content instead of uploading:")
    pasted_csv = st.text_area("Paste CSV (optional)", height=120)

# ---------------------------
# Load data
# ---------------------------
df = None
errors = []

try:
    if uploaded is not None:
        df = pd.read_csv(uploaded)
    elif pasted_csv.strip():
        df = pd.read_csv(StringIO(pasted_csv))
except Exception as e:
    errors.append(f"CSV parse error: {e}")

if df is None:
    st.info("➡️ Upload a CSV or paste CSV text in the sidebar to begin.", icon="ℹ️")
    if errors:
        st.error("\n".join(errors))
    st.stop()

# Check timestamp column exists
if time_col not in df.columns:
    st.error(f"Timestamp column '{time_col}' not found. Available columns: {list(df.columns)}")
    st.stop()

# ---------------------------
# Parse timestamps & clean
# ---------------------------
try:
    # If tz provided, treat input as UTC and convert to target tz
    ts = pd.to_datetime(df[time_col], errors="coerce", utc=bool(tz))
    if tz:
        ts = ts.dt.tz_convert(tz)
    df[time_col] = ts
except Exception as e:
    st.error(f"Failed to parse/convert timestamps: {e}")
    st.stop()

df = df.dropna(subset=[time_col]).copy()
df = df.sort_values(time_col).drop_duplicates(subset=[time_col])

if df.empty:
    st.error("No valid rows after parsing timestamps and dropping NaNs.")
    st.stop()

# --------------------------------
# NEW: Select the meter column(s)
# --------------------------------
# Suggest numeric columns only for meters
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

# Default to 'value' if present, else first numeric column (if any)
default_meter = "value" if "value" in numeric_cols else (numeric_cols[0] if numeric_cols else None)

if not numeric_cols:
    st.error(
        "No numeric columns found for meters. "
        "Ensure your CSV has at least one numeric cumulative meter column."
    )
    st.stop()

with st.sidebar:
    meter_col = st.multiselect(
        "Cumulative meter columns",
        options=numeric_cols,  # Allow selection of numeric columns only
        default=[default_meter],  # Pre-select one column by default
        help="Select one or more columns to visualize cumulative meter data."
    )

# ---------------------------
# Helper: cumulative → per-period consumption
# ---------------------------
def consumption_by_rule(dataframe: pd.DataFrame, rule: str) -> pd.Series:
    """
    Resample to 'rule' (e.g., 'H','D','M'), take last reading, diff to get period usage.
    Drops NaN last readings; optionally drops negative diffs (resets).
    """
    last_vals = (
        dataframe.set_index(time_col)[meter_col]
                 .resample(rule)
                 .last()
                 .dropna()
    )
    diffs = last_vals.diff()
    if drop_neg:
        diffs = diffs.where(diffs >= 0)
    return diffs.dropna()

# Precompute for Month view (daily) and Year view (monthly), and Day view (hourly)
daily_all   = (consumption_by_rule(df, "D") / unit_divisor)      # for Month
monthly_all = (consumption_by_rule(df, "M") / unit_divisor)      # for Year
hourly_all  = (consumption_by_rule(df, "H") / unit_divisor)      # for Day
yearly_all  = (consumption_by_rule(df, "Y") / unit_divisor)  # Resample to yearly consumption

# If all are empty, nothing to display
if daily_all.empty and monthly_all.empty and hourly_all.empty:
    st.warning("No consumptions could be computed. Check that values are cumulative and timestamps are valid.")
    st.stop()

# If yearly data is empty, warn the user
if yearly_all.empty:
    st.warning("No yearly data available.")
    st.stop()

# Annotate with parts for selectors
def add_parts(series) -> pd.DataFrame:
    df_ = series.copy()
    # pandas handles tz-aware indices for these accessors
    df_["year"]  = df_.index.year
    df_["month"] = df_.index.month
    df_["day"]   = df_.index.day
    df_["date"]  = df_.index.normalize()
    # hour only exists for hourly resample; use getattr pattern safely
    try:
        df_["hour"] = df_.index.hour
    except Exception:
        df_["hour"] = np.nan
    return df_

daily_df   = add_parts(daily_all)
monthly_df = add_parts(monthly_all)
hourly_df  = add_parts(hourly_all)
yearly_df  = add_parts(yearly_all)

# ---------------------------
# Granularity selector
# ---------------------------
gran = st.radio("Bar graph by", ["Day", "Month", "Year", "Years"], index=1, horizontal=True)

# ---------------------------
# MONTH view → daily bars for selected year+month
# ---------------------------
if gran == "Month":
    if daily_df.empty:
        st.warning("No daily data available.")
        st.stop()

    years = sorted(daily_df["year"].unique())
    col1, col2, col3 = st.columns([1,1,3])
    with col1:
        sel_year = st.selectbox("Year", years, index=len(years)-1)
    with col2:
        months_avail = sorted(daily_df.loc[daily_df["year"] == sel_year, "month"].unique())
        sel_month = st.selectbox(
            "Month",
            months_avail,
            index=0,
            format_func=lambda m: pd.Timestamp(year=sel_year, month=m, day=1).strftime("%B")
        )

    month_mask = (daily_df["year"] == sel_year) & (daily_df["month"] == sel_month)
    month_view = daily_df.loc[month_mask, meter_col + ["date", "day"]].copy().sort_values("day")

    if month_view.empty:
        st.warning(f"No data for {sel_year}-{sel_month:02d}.")
        st.stop()

    title_txt = f"Daily Consumption – {sel_year}-{sel_month:02d}"
    y_title = f"Daily consumption [{unit_label}]"

    chart = (
        alt.Chart(month_view).transform_fold(meter_col, as_=["series", "value"])
        .mark_bar(color="#4C78A8")
        .encode(
            x=alt.X("day:O", title="Day of month"),
            y=alt.Y("value:Q", title=y_title),
            color="series:N",
            tooltip=[
                alt.Tooltip("date:T", title="Date"),
                alt.Tooltip("daily:Q", title=y_title, format=".2f"),
            ],
        )
        .properties(height=420, title=title_txt)
    )
    st.altair_chart(chart, use_container_width=True)


# ---------------------------
# YEAR view → monthly bars for selected year
# ---------------------------
elif gran == "Year":
    if monthly_df.empty:
        st.warning("No monthly data available.")
        st.stop()

    years = sorted(monthly_df["year"].unique())
    sel_year = st.selectbox("Year", years, index=len(years)-1)

    ymask = monthly_df["year"] == sel_year
    year_view = monthly_df.loc[ymask, meter_col + ["month"]].copy().sort_values("month")
    if year_view.empty:
        st.warning(f"No data for {sel_year}.")
        st.stop()

    # Add month labels for nicer x-axis
    year_view["month_name"] = year_view["month"].apply(lambda m: pd.Timestamp(sel_year, m, 1).strftime("%b"))

    title_txt = f"Monthly Consumption – {sel_year}"
    y_title = f"Monthly consumption [{unit_label}]"

    chart = (
        alt.Chart(year_view).transform_fold(meter_col, as_=["series", "value"])
        .mark_bar(color="#59A14F")
        .encode(
            x=alt.X(
                "month_name:O",
                title="Month",
                sort=["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
            ),
            y=alt.Y("value:Q", title=y_title),
            color="series:N",
            tooltip=[
                alt.Tooltip("month:O", title="Month (number)"),
                alt.Tooltip("monthly:Q", title=y_title, format=".2f"),
            ],
        )
        .properties(height=420, title=title_txt)
    )

    st.altair_chart(chart, use_container_width=True)

# ---------------------------
# Years → Yearly bars for all available years
# ---------------------------
elif gran == "Years":
    if yearly_df.empty:
        st.warning("No yearly data available.")
        st.stop()

    # Aggregate the yearly data
    year_view = yearly_df.copy().sort_values("year")
    if year_view.empty:
        st.warning("No yearly data available.")
        st.stop()

    # "Year" and "Yearly consumption" as column labels
    title_txt = "Yearly Overview – Energy Consumption"
    y_title = f"Yearly consumption [{unit_label}]"

    # Create the bar chart using Altair
    chart = (
        alt.Chart(year_view).transform_fold(meter_col, as_=["series", "value"])
        .mark_bar(color="#FF7F0E")
        .encode(
            x=alt.X("year:O", title="Year"),
            y=alt.Y("value:Q", title=y_title),
            color="series:N",
            tooltip=[
                alt.Tooltip("year:O", title="Year"),
                alt.Tooltip("yearly:Q", title=y_title, format=".2f"),
            ],
        )
        .properties(height=420, title=title_txt)
    )

    # Render the chart
    st.altair_chart(chart, use_container_width=True)

# ---------------------------
# DAY view → hourly bars for selected date
# ---------------------------
else:
    if hourly_df.empty:
        st.warning("No hourly data available.")
        st.stop()

    # Build cascading selectors: Year -> Month -> Day (only days that exist in hourly_df)
    years = sorted(hourly_df["year"].unique())
    c1, c2, c3, c4 = st.columns([1,1,1,3])

    with c1:
        sel_year = st.selectbox("Year", years, index=len(years)-1)

    with c2:
        months_avail = sorted(hourly_df.loc[hourly_df["year"] == sel_year, "month"].unique())
        sel_month = st.selectbox(
            "Month",
            months_avail,
            index=0,
            format_func=lambda m: pd.Timestamp(sel_year, m, 1).strftime("%B")
        )

    with c3:
        days_avail = sorted(hourly_df.loc[
            (hourly_df["year"] == sel_year) & (hourly_df["month"] == sel_month), "day"
        ].unique())
        if not days_avail:
            st.warning(f"No days with hourly data for {sel_year}-{sel_month:02d}.")
            st.stop()
        sel_day = st.selectbox("Day", days_avail, index=0)

    dmask = (
        (hourly_df["year"] == sel_year)
        & (hourly_df["month"] == sel_month)
        & (hourly_df["day"] == sel_day)
    )
    day_view = hourly_df.loc[dmask, meter_col + ["hour"]].copy().sort_values("hour")

    if day_view.empty:
        st.warning(f"No hourly data for {sel_year}-{sel_month:02d}-{sel_day:02d}.")
        st.stop()

    title_txt = f"Hourly Consumption – {sel_year}-{sel_month:02d}-{sel_day:02d}"
    y_title = f"Hourly consumption [{unit_label}]"

    chart = (
        alt.Chart(day_view).transform_fold(meter_col, as_=["series", "value"])
        .mark_bar(color="#F28E2B")
        .encode(
            x=alt.X("hour:O", title="Hour"),
            y=alt.Y("value:Q", title=y_title),
            color="series:N",
            tooltip=[
                alt.Tooltip("hour:Q", title="Hour"),
                alt.Tooltip("hourly:Q", title=y_title, format=".2f"),
            ],
        )
        .properties(height=420, title=title_txt)
    )
    st.altair_chart(chart, use_container_width=True)

