Book Recommender (Cluster-Based)
Goal: Build a simple, explainable recommender that suggests similar books using content-based clustering (no ratings).

Data Sources (≥1,000 books total)
Project Gutenberg (web scraping)

From the categories page, we fetched main_category and subcategory, then collected per-subcategory book listings.
Extracted fields: title, author, url, plus the two categories.
Polite scraping: custom User-Agent + short randomized sleeps; pagination via start_index.
Targeted ≥500 unique books (oversampled then deduped by URL).
Open Library (API)

Mirrored Gutenberg’s (main_category, subcategory) pairs.
Queried Subjects API first; fallback to Search by subject and generic Search.
Mapped results to the same schema: title, author, url, main_category, subcategory.
Targeted ≥500 unique books (again oversampled + deduped).
Outputs: gutenberg_books.csv, openlibrary_books.csv → merged into books_merged.csv.

Pipeline
Gutenberg scrape

Discover categories → shuffle subcategories to avoid alphabetical bias → cap max items per subcategory.
Parse listing cards (.booklink) or fallback <li><a> to get title, author, url.
Deduplicate by url.
Open Library collect

Convert subcategory to subject slug (e.g., “Science Fiction” → science-fiction).
Pull works via /subjects/{slug}.json; if sparse, use /search.json?subject=… or q=….
Attach the originating (main_category, subcategory) to keep schema aligned.
Deduplicate by url.
Merge & clean

Concatenate, add source flag.
Drop exact duplicate urls and near-duplicates by normalized (title, author).
Modeling (Clustering Recommender)
Text corpus per book: title + author + main_category + subcategory
(categories lightly up-weighted).
Vectorization: TF-IDF (1–2 grams, English stop words).
Dimensionality reduction: TruncatedSVD (e.g., 50 components).
Clustering: KMeans (K tunable).
Recommendations: cosine similarity within the same cluster (nearest neighbors).
App (Streamlit)
Loads books_merged.csv (or lets you upload a CSV).
Sidebar controls:
Filter by categories.
Tune K, SVD components, TF-IDF max_features, category weighting.
Pick a seed book → get top-N within-cluster recommendations.
Visualizes clusters in 2D (PCA projection).
Lets you download the data with cluster labels.
Files
gutenberg_books.csv — scraped metadata (≥500)
openlibrary_books.csv — API metadata (≥500)
books_merged.csv — merged & deduped dataset (input to the app)
app.py — Streamlit clustering recommender
notebooks/ — scraping, API, merge steps
One-liner: We combined Gutenberg scraping and Open Library APIs to build a ~1k-book dataset, then used TF-IDF → SVD → KMeans to cluster and recommend books, wrapped in an interactive Streamlit app.

Run:

pip install requests beautifulsoup4 pandas scikit-learn plotly streamlit
streamlit run app.py
