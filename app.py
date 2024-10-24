import io
import json
import re
from datetime import datetime
from typing import Any, Callable

import matplotlib as mpl
import numpy as np
import pandas as pd
import plotly.express as px
import requests
import streamlit as st
from pandas.api.types import is_numeric_dtype
from pandas.api.typing import NaTType
from pandas.io.formats.style import Styler

st.set_page_config(page_title="CH2 Log Parser", layout="wide")


def convert_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Convert the dtypes of the dataframe columns to the correct types."""
    new_df = pd.DataFrame(df)
    for col, dtype in df.dtypes.items():
        if dtype == "object":
            if col.endswith("Time"):
                new_df[col] = pd.to_timedelta(df[col])
            elif col.endswith("Ratio"):
                new_df[col] = df[col].astype("string").str.replace("%", "").astype("float") / 100
        elif is_numeric_dtype(dtype):
            if col.endswith("Count"):
                new_df[col] = df[col].fillna(0).astype("int")
        else:
            new_df[col] = df[col]
    return new_df


def human_td_format(td: pd.Timedelta | NaTType) -> str | None:
    """Format a timedelta object to a human-readable string."""
    if pd.isna(td):
        return

    td_ = td.round("ms").components
    s = f"{td_.hours:02}:{td_.minutes:02}:{td_.seconds:02}.{td_.milliseconds:03}"
    if td_.days:
        s = f"{td_.days} days, {s}"
    return s


def seconds_td_format(td: pd.Timedelta | NaTType) -> str | None:
    """Format a timedelta object as the total number of seconds."""
    if not pd.isna(td):
        return f"{td.total_seconds():.2f}"


def create_formatter(df: pd.DataFrame) -> dict[str, Callable[[Any], str]]:
    """Create a dictionary of formatters for the dataframe."""
    formatter = {}
    for col, dtype in df.dtypes.items():
        if dtype == "timedelta64[ns]":
            formatter[col] = (
                human_td_format
                if st.session_state.get("duration_format", "HH:MM:SS") == "HH:MM:SS"
                else seconds_td_format
            )
        elif col.endswith("Ratio"):
            formatter[col] = lambda x: f"{x * 100:.2f}%"
        elif str(dtype).lower().startswith("int"):
            formatter[col] = lambda x: f"{x:,}"
        elif str(dtype).lower().startswith("float"):
            formatter[col] = lambda x: f"{x:,.2f}"
    return formatter


def geo_mean(s: pd.Series) -> np.float64:
    """Calculate the geometric mean of a series."""
    return np.exp(np.log(s.dt.total_seconds()).mean())


def qph(s: pd.Series) -> np.float64:
    """Calculate the queries per hour from a series of query times."""
    return 3600 / s.dt.total_seconds().mean()


def count_non_zero(s: pd.Series) -> int:
    """Count number of non-zero values in a numeric series."""
    return s.astype(bool).sum()


def create_by_loop_summary_df(
    base_df: pd.DataFrame,
    agg: dict[str, str | Callable[[pd.Series], Any]] | None = None,
    formatter: str | dict[str, str] | None = None,
) -> Styler:
    """Create a dataframe for the summary section of the app.

    Returns a Styler object for the summary dataframe, that can be passed to `st.dataframe()`.
    """
    summary_df = (
        base_df.groupby(["file", "loop"])
        .agg(agg)
        .stack()
        .unstack(level=0)
        .reset_index(level=1, drop=True)
    )[file_labels["label"].to_list()]
    return summary_df.style.format(formatter or create_formatter(summary_df))


def create_summary_section(section_name: str, summary_dfs: dict[str, Styler]):
    """Render a "summary" section with multiple dataframes."""
    if not summary_dfs:
        return

    st.subheader(section_name)
    with st.container():
        cols = st.columns(len(summary_dfs))

        for c, (name, sdf) in zip(cols, summary_dfs.items()):
            with c:
                st.write(f"#### {name}")
                st.dataframe(sdf)


def background_with_norm(
    cmap: str, vmin: float, vcenter: float, vmax: float
) -> Callable[[pd.Series], list[str]]:
    """Create a background color function that maps values to a colormap with a center value."""
    cmap = mpl.colormaps[cmap]
    norm = mpl.colors.TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
    return lambda s: [
        f"color:black;background-color: {mpl.colors.to_hex(c.flatten()):s}"
        for c in cmap(norm(s.values))
    ]


def percent_diff(x: pd.Series, y: pd.Series) -> pd.Series:
    """Calculate element-wise percentage difference between two series."""
    return (y - x) / x


def percent_points_diff(x: pd.Series, y: pd.Series) -> pd.Series:
    """Calculate element-wise percentage point difference between two series.

    Assumes that input series represent proportions and have values in [0, 1].
    """
    return (y - x) * 100


st.title("CH2 Log Parser")

if "web_files" not in st.session_state:
    st.session_state.web_files = []

with st.expander("**File Upload and Labeling**", expanded=True):
    c1, c2 = st.columns(2)
    with c1:
        uploaded_files = st.file_uploader(
            "Upload CH2 log files",
            type=["log", "txt"],
            accept_multiple_files=True,
        )

        st.write("Or fetch from Perf Jenkins")
        with st.form("web_file_form"):
            fc1, fc2 = st.columns([0.7, 0.3])
            with fc1:
                job = st.text_input("Job name", "ColumnarAWSWeekly")
            with fc2:
                build = st.number_input("Build number", min_value=1, step=1)
            submitted = st.form_submit_button("Add log from Jenkins")
            if submitted:
                url = f"https://perf.jenkins.couchbase.com/job/{job}/{build}/artifact/ch2_analytics.log"
                resp = requests.get(url)
                if resp.ok:
                    file = io.BytesIO(resp.content)
                    file.name = f"{job}/{build}"
                    st.session_state.web_files.append(file)
                else:
                    st.error(f"Failed to fetch log from {url}")

        c3, c4 = st.columns([0.9, 0.1])
        for file in st.session_state.web_files:
            with c3:
                st.write(file.name)
            if c4.button(label="", icon=":material/close:", key=f"remove_{id(file)}"):
                st.session_state.web_files.remove(file)
                st.rerun()

    if not (all_files := uploaded_files + st.session_state.web_files):
        st.stop()

    with c2:
        st.caption("File Label Editor")
        file_label_df = pd.DataFrame(
            [{"file": file.name, "label": file.name} for file in all_files]
        )
        file_labels = st.data_editor(
            file_label_df,
            disabled=["file"],
            hide_index=True,
            column_config={
                "label": st.column_config.TextColumn(
                    max_chars=50, required=True, validate=r".*\S.*"
                )
            },
            use_container_width=True,
        )
        if file_labels["label"].unique().size != file_labels.shape[0]:
            st.error("Labels must be unique.")
            st.stop()

ts_pattern = r"(\d{2})-(\d{2})-(\d{4}) (\d{2}):(\d{2}):(\d{2})"
query_pattern = r"AClient (\d+):Loop (\d+):Q(\d{2})"
pattern = re.compile(rf"{ts_pattern}.*{query_pattern}:\s+([\w\s]+):\s+(.*)")

start_times = {}
end_times = {}
rows = []
for i, file in enumerate(all_files):
    file.seek(0)
    for line in file:
        if m := pattern.match(line.decode()):
            mm, dd, yyyy, h, m, s, client, loop, query, action, data = m.groups()
            timestamp = datetime(
                year=int(yyyy),
                month=int(mm),
                day=int(dd),
                hour=int(h),
                minute=int(m),
                second=int(s),
            )
            if (action := action.lower()) == "metrics":
                try:
                    data_parsed = json.loads(data.replace("'", '"'))
                except json.JSONDecodeError:
                    data_parsed = {}

                rows.append(
                    {
                        "file": file_labels.iloc[i]["label"],
                        "loop": int(loop),
                        "query": int(query),
                        "client": int(client),
                        "start_time": start_times.get((file.name, loop, query, client)),
                        "end_time": end_times.get((file.name, loop, query, client)),
                        **data_parsed,
                    }
                )
            elif action.startswith("started"):
                start_times[(file.name, loop, query, client)] = timestamp
            elif action.startswith("ended"):
                end_times[(file.name, loop, query, client)] = timestamp

df = convert_dtypes(pd.DataFrame(rows))
df.set_index(["file", "loop", "query", "client"], inplace=True)

st.header("Summary")
st.warning("Values will not be accurate if # clients > 1", icon="⚠️")

c1, c2, _ = st.columns([0.2, 0.2, 0.6])
with c1:
    time_field = st.radio(
        "Field used for query time summaries:",
        ["elapsedTime", "executionTime"],
    )
with c2:
    st.session_state.duration_format = st.radio(
        "Duration format:",
        ["HH:MM:SS", "seconds"],
    )

create_summary_section(
    "Query performance",
    {
        "Geo-mean query time (s)": create_by_loop_summary_df(df, {time_field: geo_mean}),
        "Query set time": create_by_loop_summary_df(df, {time_field: "sum"}),
        "QPH": create_by_loop_summary_df(df, {time_field: qph}),
    },
)

create_summary_section(
    "Buffer Cache Hit Ratio",
    {
        "Minimum": create_by_loop_summary_df(df, {"bufferCacheHitRatio": "min"}, "{:.2%}"),
        "Maximum": create_by_loop_summary_df(df, {"bufferCacheHitRatio": "max"}, "{:.2%}"),
        "Median": create_by_loop_summary_df(df, {"bufferCacheHitRatio": "median"}, "{:.2%}"),
    },
)

if any(col.startswith("remoteStorage") for col in df.columns):
    create_summary_section(
        "Cloud Storage Reads",
        {
            "Queries accessing cloud storage": create_by_loop_summary_df(
                df, {"remoteStorageRequestsCount": count_non_zero}, "{} of 22"
            ),
            "Cloud storage requests": create_by_loop_summary_df(
                df, {"remoteStorageRequestsCount": "sum"}
            ),
            "Cloud storage pages read": create_by_loop_summary_df(
                df, {"remoteStoragePagesReadCount": "sum"}
            ),
            "Cloud storage pages persisted": create_by_loop_summary_df(
                df, {"remoteStoragePagesPersistedCount": "sum"}
            ),
        },
    )

if len(file_labels) > 1:
    st.divider()
    st.header("Query-by-Query Comparison")
    c1, c2, _ = st.columns([0.2, 0.2, 0.6])
    with c1:
        file_A = st.selectbox("First log file", file_labels["label"])

    with c2:
        file_B = st.selectbox(
            "Second log file", file_labels.loc[file_labels["label"] != file_A]["label"]
        )

    qbq_comparison_cols = {
        col: (percent_diff, "{:+.2%}", background_with_norm("RdYlGn_r", -1, 0, 5))
        if not col.endswith("Ratio")
        else (
            percent_points_diff,
            {file_A: "{:.2%}", file_B: "{:.2%}", "diff": "{:+.2}"},
            background_with_norm("RdYlGn_r", -100, 0, 100),
        )
        for col in df.columns
        if col not in ["start_time", "end_time"]
    }

    comparison_col = st.selectbox("Comparison column", qbq_comparison_cols.keys())
    diff_func, diff_fmt, diff_background = qbq_comparison_cols[comparison_col]

    qbq_df = (
        df[comparison_col]
        .to_frame()
        .stack()
        .unstack(level=0)
        .reset_index(level=3, drop=True)[[file_A, file_B]]
    )
    qbq_df["diff"] = diff_func(qbq_df[file_A], qbq_df[file_B])
    qbq_formatter = create_formatter(qbq_df) | (
        {"diff": diff_fmt} if isinstance(diff_fmt, str) else diff_fmt
    )

    qbq_df_st_kwargs = {"width": 800, "height": 820}
    st.dataframe(
        qbq_df.style.format(qbq_formatter).apply(diff_background, subset=["diff"], axis=0),
        width=800,
        height=820,
    )

st.divider()
st.header("Query Timeline")
timeline_df = df[["start_time", "end_time"]]
timeline_df["client"] = timeline_df.index.get_level_values("client").astype("string")
timeline_df["query"] = timeline_df.index.get_level_values("query").astype("string")
for name in timeline_df.index.get_level_values("file").unique():
    st.subheader(name)
    timeline_fig = px.timeline(
        timeline_df.loc[name],
        x_start="start_time",
        x_end="end_time",
        y="client",
        color="query",
        category_orders={"client": sorted(timeline_df.index.get_level_values("client").unique())},
        color_discrete_map=dict(
            zip(sorted(timeline_df["query"].unique()), px.colors.qualitative.Light24)
        ),
    )
    st.plotly_chart(timeline_fig, use_container_width=True)

st.divider()
st.header("Raw Data")
for name in df.index.get_level_values("file").unique():
    st.subheader(name)
    st.dataframe(df.loc[name].style.format(create_formatter(df)))
