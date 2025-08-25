import os
import re
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Book Recommender (Clusters)", layout="wide")
st.title("Book Recommender — Cluster-based")


@st.cache_data(show_spinner=False)
def load_data(path):
    df = pd.read_csv(path)
    rename = {c: c.lower() for c in df.columns}
    df = df.rename(columns=rename)
    keep = ["main_category","subcategory","title","author","url"]
    df = df[[c for c in keep if c in df.columns]].copy()
    for c in keep:
        if c in df: df[c] = df[c].fillna("")
    if "url" in df:
        df = df.drop_duplicates(subset=["url"]).reset_index(drop=True)
    return df

def build_corpus(df, cat_weight=2):
    def norm(s):
        s = str(s)
        s = re.sub(r"\s+", " ", s).strip()
        return s
    title = df["title"].map(norm)
    author = df["author"].map(norm)
    mainc = df["main_category"].map(norm)
    subc  = df["subcategory"].map(norm)
    cats = (mainc + " " + subc).str.strip()
    cats_weighted = (cats + " ") * cat_weight
    return (title + " | " + author + " | " + cats_weighted.str.strip()).str.lower()

@st.cache_data(show_spinner=True)
def embed(df, max_features=5000, n_components=50, cat_weight=2):
    corpus = build_corpus(df, cat_weight=cat_weight)
    vec = TfidfVectorizer(max_features=max_features, ngram_range=(1,2), min_df=2, stop_words="english")
    X = vec.fit_transform(corpus)
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    Z = svd.fit_transform(X)
    return Z

@st.cache_data(show_spinner=False)
def cluster_embeddings(Z, k):
    km = KMeans(n_clusters=k, n_init="auto", random_state=42)
    labels = km.fit_predict(Z)
    centers = km.cluster_centers_
    return labels, centers

def get_recommendations(idx, Z, labels, topn=10):
    c = labels[idx]
    same = np.where(labels == c)[0]
    same = same[same != idx]
    if same.size == 0: return []
    sims = cosine_similarity(Z[idx].reshape(1, -1), Z[same]).flatten()
    order = np.argsort(-sims)[:topn]
    return list(zip(same[order], sims[order]))

def pca2(Z):
    p = PCA(n_components=2, random_state=42)
    return p.fit_transform(Z)


st.sidebar.header("Settings")
default_path = os.environ.get("BOOKS_CSV", "./books_merged.csv")

mode = st.sidebar.radio("Data source", ["Path", "Upload"], index=0)
if mode == "Path":
    data_path = st.sidebar.text_input("CSV path", value=default_path)
    if not data_path: st.stop()
    df = load_data(data_path)
else:
    upl = st.sidebar.file_uploader("Upload merged CSV", type=["csv"])
    if not upl: st.stop()
    df = load_data(upl)

st.sidebar.success(f"Loaded {len(df)} books")

main_filter = st.sidebar.multiselect("Filter main_category", sorted(df["main_category"].unique()))
if main_filter:
    df = df[df["main_category"].isin(main_filter)].reset_index(drop=True)

sub_filter = st.sidebar.multiselect("Filter subcategory", sorted(df["subcategory"].unique()))
if sub_filter:
    df = df[df["subcategory"].isin(sub_filter)].reset_index(drop=True)

if len(df) < 5:
    st.warning("Not enough books after filtering.")
    st.stop()

k_default = max(6, int(np.sqrt(len(df) / 2)))
k = st.sidebar.slider("Clusters (K)", 4, 50, k_default, 1)
cat_weight = st.sidebar.slider("Category weighting", 1, 5, 2)
max_features = st.sidebar.select_slider("TF-IDF max_features", options=[3000, 5000, 8000, 12000], value=5000)
n_components = st.sidebar.select_slider("SVD components", options=[25, 50, 75, 100], value=50)


with st.spinner("Embedding & clustering…"):
    Z = embed(df, max_features=max_features, n_components=n_components, cat_weight=cat_weight)
labels, centers = cluster_embeddings(Z, k)

df_view = df.copy()
df_view["cluster"] = labels


st.subheader("Pick a book to get recommendations")
sel_title = st.selectbox("Book", options=df_view["title"], index=0)
sel_idx = df_view.index[df_view["title"] == sel_title][0]

topn = st.slider("How many recommendations?", 5, 30, 10)
recs = get_recommendations(sel_idx, Z, labels, topn=topn)

rec_rows = []
for ridx, sim in recs:
    rec_rows.append({
        "title": df_view.loc[ridx, "title"],
        "author": df_view.loc[ridx, "author"],
        "main_category": df_view.loc[ridx, "main_category"],
        "subcategory": df_view.loc[ridx, "subcategory"],
        "similarity": float(sim),
        "url": df_view.loc[ridx, "url"]
    })
rec_df = pd.DataFrame(rec_rows)

left, right = st.columns([0.55, 0.45])

with left:
    st.markdown(f"**Selected:** {df_view.loc[sel_idx, 'title']} — {df_view.loc[sel_idx, 'author']}")
    st.dataframe(
        rec_df,
        use_container_width=True,
        column_config={
            "url": st.column_config.LinkColumn("url", display_text="Open"),
            "similarity": st.column_config.NumberColumn(format="%.3f")
        }
    )

with right:
    st.markdown("**Cluster map (2D)**")
    pts2 = pca2(Z)
    plot_df = pd.DataFrame({
        "x": pts2[:,0],
        "y": pts2[:,1],
        "cluster": labels.astype(int),
        "title": df_view["title"],
        "author": df_view["author"]
    })
    fig = px.scatter(plot_df, x="x", y="y", color="cluster", hover_data=["title","author"], height=520)
    fig.add_scatter(x=[pts2[sel_idx,0]], y=[pts2[sel_idx,1]], mode="markers",
                    marker=dict(size=14, line=dict(width=2), symbol="star"), name="selected")
    st.plotly_chart(fig, use_container_width=True, theme="streamlit")

st.subheader("Cluster sizes")
sizes = (df_view.groupby("cluster").size().reset_index(name="count").sort_values("count", ascending=False))
st.dataframe(sizes, use_container_width=True)

st.download_button("Download CSV with cluster labels", data=df_view.to_csv(index=False).encode("utf-8"),
                   file_name="books_with_clusters.csv", mime="text/csv")