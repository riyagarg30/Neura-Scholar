import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import ast, json, glob, os
from pathlib import Path
from typing import List, Dict, Any, Optional

# ─── Page config ────────────────────────────────────────
st.set_page_config(
    page_title="Data Quality Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Hash lists so cache_data can handle list-columns ────
hash_funcs = { list: lambda x: tuple(x) }

# ─── Helper Functions ────────────────────────────────────
def front_cols(df: pd.DataFrame) -> List[str]:
    return [c for c in ('id','submitter','title','created_date') if c in df.columns]

@st.cache_data
def load_raw_file(path: str) -> pd.DataFrame:
    ext = Path(path).suffix.lower()
    if ext == ".csv":
        return pd.read_csv(path, low_memory=False)
    elif ext == ".tsv":
        return pd.read_csv(path, sep="\t", low_memory=False)
    elif ext == ".txt":
        lines = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                txt = line.strip()
                if txt:
                    lines.append(txt)
        return pd.DataFrame({"query": lines})
    elif ext == ".json":
        # generic dict->DF
        data = json.load(open(path, 'r', encoding='utf-8'))
        return pd.DataFrame({"key": list(data.keys()), "value": list(data.values())})
    else:
        raise ValueError(f"Unsupported file type: {ext}")

@st.cache_data(hash_funcs=hash_funcs)
def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # date
    if 'created_yymm' in df.columns:
        df['created_yymm'] = df['created_yymm'].astype(str).str.zfill(4)
        df['created_date'] = pd.to_datetime(df['created_yymm'], format='%y%m', errors='coerce')
    else:
        df['created_date'] = pd.NaT
    # categories
    if 'categories' in df.columns:
        df['categories'] = (
            df['categories']
            .apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else ([x] if pd.notna(x) else []))
            .apply(tuple)
        )
    else:
        df['categories'] = [tuple()]*len(df)
    # abstract length
    df['abstract_length'] = df['abstract'].str.len() if 'abstract' in df.columns else 0
    return df

@st.cache_data
def compute_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    m: Dict[str, Any] = {}
    # time series
    if not df['created_date'].isna().all():
        td = (df.dropna(subset=['created_date'])
                .groupby(df['created_date'].dt.to_period('M'))
                .size()
                .reset_index(name='count'))
        td['created_date'] = td['created_date'].dt.to_timestamp()
        m['time_df'] = td
    else:
        m['time_df'] = pd.DataFrame(columns=['created_date','count'])
    # missing
    miss = df.isna().mean().sort_values(ascending=False).reset_index()
    miss.columns = ['column','missing_pct']
    m['missing'] = miss
    # duplicates
    drop_cols = [c for c in df.columns if df[c].apply(lambda x: isinstance(x,(list,tuple))).any()]
    m['dup_pct'] = df.drop(columns=drop_cols).duplicated().mean()*100
    # top submitters
    if 'submitter' in df.columns:
        top = df['submitter'].value_counts().head(10).reset_index()
        top.columns = ['submitter','count']
        m['top_sub'] = top
    else:
        m['top_sub'] = pd.DataFrame(columns=['submitter','count'])
    # core metrics
    m['record_count'] = len(df)
    m['submitter_count'] = df.get('submitter', pd.Series()).nunique()
    m['avg_abstract_length'] = df['abstract_length'].mean()
    m['missing_pct'] = df.isna().mean().mean()*100
    m['preview'] = df.head(100)
    return m

@st.cache_data(hash_funcs=hash_funcs)
def filter_dataframe(
    df: pd.DataFrame,
    categories: Optional[List[str]] = None,
    versions: Optional[List[str]] = None
) -> pd.DataFrame:
    fd = df.copy()
    if categories and 'categories' in df.columns:
        fd = fd[fd['categories'].apply(lambda cs: any(c in categories for c in cs))]
    if versions and 'latest_version' in df.columns:
        fd = fd[fd['latest_version'].isin(versions)]
    return fd

# ─── Discover files ─────────────────────────────────────
DATA_DIR = os.path.expanduser("/mnt/object/dashboard-files")
all_paths = sum((glob.glob(os.path.join(DATA_DIR, ext)) for ext in ("*.csv","*.tsv","*.txt","*.json")), [])
all_names = [os.path.basename(p) for p in all_paths]

st.sidebar.header("Choose dataset")
if not all_names:
    st.error(f"No files in {DATA_DIR}")
    st.stop()

sel_name = st.sidebar.selectbox("File", all_names)
sel_path = all_paths[all_names.index(sel_name)]
ext = Path(sel_path).suffix.lower()

# ─── all_lists.json view ────────────────────────────────
if sel_name == "all_lists.json" and ext == ".json":
    lists = json.load(open(sel_path, 'r', encoding='utf-8'))
    st.title(f"📂 List Overview — {sel_name}")
    counts = {k: len(v) for k,v in lists.items()}
    df_counts = pd.DataFrame.from_dict(counts, orient="index", columns=["count"]) \
                          .reset_index().rename(columns={"index":"list_name"})
    # metrics
    cols = st.columns(len(counts))
    for col, (name, cnt) in zip(cols, counts.items()):
        col.metric(name.replace("_"," ").title(), cnt)
    # bar chart
    st.subheader("Comparison of List Sizes")
    st.bar_chart(df_counts.set_index("list_name")["count"])
    # previews
    st.subheader("List Previews")
    for name, items in lists.items():
        with st.expander(f"{name} ({len(items)} items)"):
            st.write("\n".join(items))
    st.stop()

# ─── all_buckets_citations.json view ────────────────────
if sel_name == "all_buckets_citations.json" and ext == ".json":
    raw = load_raw_file(sel_path)
    df = pd.DataFrame({"paper_id": raw["key"], "citations": raw["value"]})
    df["citations_count"] = df["citations"].apply(len)
    st.title(f"📑 Citations Dashboard — {sel_name}")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Papers", len(df))
    c2.metric("Total Links", df["citations_count"].sum())
    c3.metric("Avg per Paper", f"{df['citations_count'].mean():.1f}")
    c4.metric("Max Citations", int(df["citations_count"].max()))
    st.subheader("Top 20 Papers by Citations")
    st.bar_chart(df.nlargest(20,"citations_count").set_index("paper_id")["citations_count"])
    st.subheader("Citation Distribution")
    fig = px.histogram(df, x="citations_count", nbins=30, marginal="box")
    st.plotly_chart(fig, use_container_width=True)
    st.subheader("Sample Records")
    st.dataframe(df.head(100), height=300)
    st.stop()

# ─── queries.txt view ───────────────────────────────────
raw = load_raw_file(sel_path)
if ext == ".txt" and "query" in raw.columns:
    st.title(f"🔍 Query Dashboard — {sel_name}")
    term = st.sidebar.text_input("Search queries")
    sub = raw[raw["query"].str.contains(term, case=False)] if term else raw.copy()
    sub["word_count"] = sub["query"].str.split().str.len()
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Avg Words", f"{sub['word_count'].mean():.1f}")
    c2.metric("Max Words", int(sub["word_count"].max()))
    c3.metric("Min Words", int(sub["word_count"].min()))
    c4.metric("Total Queries", len(sub))
    st.text_area(
        "queries_text_area",
        "\n".join(sub["query"]),
        height=400,
        label_visibility="hidden"
    )
    st.stop()

# ─── CSV/TSV standard view ─────────────────────────────
df = prepare_dataframe(raw)
# sidebar filters
st.sidebar.header("Filters")
filter_applied = False
sel_cats = None
if 'categories' in df.columns:
    all_cats = sorted({c for tup in df['categories'] for c in tup})
    sel_cats = st.sidebar.multiselect("Categories", all_cats, key="cat_filter")
    if sel_cats: filter_applied = True
sel_vers = None
if 'latest_version' in df.columns:
    all_vers = sorted(df['latest_version'].dropna().unique())
    sel_vers = st.sidebar.multiselect("Versions", all_vers, key="ver_filter")
    if sel_vers: filter_applied = True
filtered_df = filter_dataframe(df, sel_cats, sel_vers) if filter_applied else df
metrics = compute_metrics(filtered_df)

# ─── Render dashboard ───────────────────────────────────
st.title(f"📊 Data Quality Dashboard — {sel_name}")
if filter_applied:
    st.info(f"Showing {len(filtered_df)}/{len(df)} records")

# high-level
st.markdown("---")
c1,c2,c3,c4 = st.columns(4)
c1.metric("Records", metrics['record_count'])
c2.metric("Submitters", metrics['submitter_count'])
c3.metric("Avg Abs.Len.", f"{metrics['avg_abstract_length']:.0f}")
c4.metric("% Missing", f"{metrics['missing_pct']:.1f}%")

# time series
st.subheader("Record Count Over Time")
if not metrics['time_df'].empty:
    st.line_chart(metrics['time_df'].set_index('created_date')['count'])
else:
    st.info("No time series data")

# missing
st.subheader("Missing % by Column")
st.bar_chart(metrics['missing'].set_index('column')['missing_pct'])

# duplicates
st.subheader("Duplicate Records")
st.write(f"{metrics['dup_pct']:.2f}% duplicates")

# submitters
st.subheader("Top Submitters")
if not metrics['top_sub'].empty:
    st.bar_chart(metrics['top_sub'].set_index('submitter')['count'])
else:
    st.info("No submitter data")

# categories
if 'categories' in filtered_df.columns:
    st.subheader("Category Counts")
    cats = pd.Series([c for tup in filtered_df['categories'] for c in tup]).value_counts()
    fig = px.bar(
        cats.reset_index().rename(columns={'index':'category', 0:'count'}),
        x='category', y='count'
    )
    st.plotly_chart(fig, use_container_width=True)

# abstract lengths
st.subheader("Abstract Length Distribution")
if filtered_df['abstract_length'].sum() > 0:
    fig2 = px.histogram(filtered_df, x='abstract_length', nbins=30, marginal="box")
    st.plotly_chart(fig2, use_container_width=True)
else:
    st.info("No abstract data")

# preview
st.subheader("Sample Records")
st.dataframe(metrics['preview'], height=300)
