import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from datetime import datetime

st.set_page_config(
    page_title="Retail Customer Segmentation",
    page_icon="🛍️",
    layout="wide"
)

st.markdown("""
<style>
    .main {background-color: #f7f9fc;}
    .block-container {padding-top: 1.5rem; padding-bottom: 2rem;}
    div[data-testid="stMetric"] {
        background: white; padding: 16px; border-radius: 12px;
        border: 1px solid #e6eaf0;
    }
    .hero {
        padding: 24px; border-radius: 16px; color: white;
        background: linear-gradient(120deg, #172554, #2563eb);
        margin-bottom: 20px;
    }
    .hero h1 {margin: 0; font-size: 2.1rem;}
    .hero p {margin: 8px 0 0 0; color: #e0e7ff;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <h1>🛍️ Retail Customer Segmentation</h1>
  <p>Machine Learning Lab Project – 2026 | Customer analytics powered by K-Means clustering</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("Project Details")
    st.write("**Student Numbers**")
    st.write("24P61A66I0")
    st.write("24P61A66I2")
    st.write("24P61A66I3")
    st.divider()
    st.subheader("Model Settings")
    n_clusters = st.slider("Number of customer groups (K)", 2, 8, 4)
    st.caption("K-Means groups customers with similar purchasing behaviour.")
    st.divider()
    st.caption("Dataset: Online Retail Dataset (Kaggle)")

st.markdown("### About the project")
st.write(
    "This application identifies customer groups from retail transaction data. "
    "It calculates Recency, Frequency and Monetary (RFM) features, scales them, "
    "and applies K-Means clustering to discover customers with similar purchasing patterns."
)

uploaded_file = st.file_uploader("Upload OnlineRetail.csv", type=["csv"])

@st.cache_data(show_spinner="Loading transaction data...")
def load_data(file_obj):
    return pd.read_csv(file_obj, encoding="ISO-8859-1", low_memory=False)

@st.cache_data(show_spinner="Preparing customer features...")
def prepare_customer_data(df):
    required = ["InvoiceNo", "Quantity", "InvoiceDate", "UnitPrice", "CustomerID"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))

    data = df.copy()
    data["InvoiceDate"] = pd.to_datetime(data["InvoiceDate"], errors="coerce", dayfirst=True)
    data["CustomerID"] = data["CustomerID"].astype("string").str.strip()
    data["Quantity"] = pd.to_numeric(data["Quantity"], errors="coerce")
    data["UnitPrice"] = pd.to_numeric(data["UnitPrice"], errors="coerce")
    data = data.dropna(subset=["InvoiceNo", "InvoiceDate", "CustomerID", "Quantity", "UnitPrice"])
    data = data[data["CustomerID"].ne("")]

    # Remove cancelled invoices and non-positive quantities/prices for this sales analysis.
    data = data[~data["InvoiceNo"].astype(str).str.startswith("C", na=False)]
    data = data[(data["Quantity"] > 0) & (data["UnitPrice"] > 0)]
    data["Sales"] = data["Quantity"] * data["UnitPrice"]

    if data.empty:
        raise ValueError("No valid sales rows remain after cleaning the dataset.")

    snapshot_date = data["InvoiceDate"].max().normalize() + pd.Timedelta(days=1)
    rfm = data.groupby("CustomerID").agg(
        Recency=("InvoiceDate", lambda dates: (snapshot_date - dates.max().normalize()).days),
        Frequency=("InvoiceNo", "nunique"),
        Monetary=("Sales", "sum"),
        TotalItems=("Quantity", "sum"),
        LastPurchase=("InvoiceDate", "max")
    ).reset_index()

    # Log-transform skewed positive features before scaling.
    feature_cols = ["Recency", "Frequency", "Monetary"]
    features = rfm[feature_cols].copy()
    features = np.log1p(features)
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    return data, rfm, scaled_features, snapshot_date

if uploaded_file is None:
    st.info("Upload your **OnlineRetail.csv** file above to start the analysis. The original dataset is not embedded in this app.")
    st.markdown("#### Dataset columns expected")
    st.code("InvoiceNo, StockCode, Description, Quantity, InvoiceDate, UnitPrice, CustomerID, Country")
    st.stop()

try:
    raw_df = load_data(uploaded_file)
    clean_df, rfm, scaled_features, snapshot_date = prepare_customer_data(raw_df)
except Exception as e:
    st.error(f"Could not process this dataset: {e}")
    st.stop()

if len(rfm) < n_clusters:
    st.warning(f"Only {len(rfm)} customers are available. Reduce K to at most {len(rfm)}.")
    st.stop()

with st.spinner("Training K-Means and assigning customer segments..."):
    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    rfm["Cluster"] = model.fit_predict(scaled_features)
    try:
        score = silhouette_score(scaled_features, rfm["Cluster"]) if n_clusters < len(rfm) else np.nan
    except Exception:
        score = np.nan

# Friendly segment names based on cluster-level average spend and frequency.
cluster_summary = rfm.groupby("Cluster").agg(
    Customers=("CustomerID", "count"),
    AvgRecency=("Recency", "mean"),
    AvgFrequency=("Frequency", "mean"),
    AvgMonetary=("Monetary", "mean"),
    TotalRevenue=("Monetary", "sum")
).reset_index()

spend_rank = cluster_summary.set_index("Cluster")["AvgMonetary"].rank(ascending=False, method="first").astype(int)
frequency_rank = cluster_summary.set_index("Cluster")["AvgFrequency"].rank(ascending=False, method="first").astype(int)
segment_names = {}
for _, row in cluster_summary.iterrows():
    c = int(row["Cluster"])
    if spend_rank[c] == 1 and frequency_rank[c] <= 2:
        name = "Champions / High Value"
    elif spend_rank[c] <= 2:
        name = "High Spenders"
    elif frequency_rank[c] == 1:
        name = "Frequent Shoppers"
    elif row["AvgRecency"] >= cluster_summary["AvgRecency"].quantile(0.70):
        name = "At-Risk / Inactive"
    else:
        name = "Regular Customers"
    segment_names[c] = name

rfm["Segment"] = rfm["Cluster"].map(segment_names)
cluster_summary["Segment"] = cluster_summary["Cluster"].map(segment_names)

# KPI row
total_customers = rfm["CustomerID"].nunique()
total_sales = clean_df["Sales"].sum()
total_orders = clean_df["InvoiceNo"].nunique()
avg_spend = rfm["Monetary"].mean()

st.markdown("### Business overview")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Customers analysed", f"{total_customers:,}")
c2.metric("Valid orders", f"{total_orders:,}")
c3.metric("Total sales", f"£{total_sales:,.2f}")
c4.metric("Average customer spend", f"£{avg_spend:,.2f}")

st.caption(f"Data snapshot date used for Recency: {snapshot_date.strftime('%d %b %Y')}. Sales are calculated as Quantity × UnitPrice.")

tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "👥 Customer Segments", "🔎 Explore Customers", "ℹ️ Methodology"])

with tab1:
    left, right = st.columns(2)
    with left:
        fig = px.bar(
            cluster_summary.sort_values("Cluster"),
            x="Segment", y="Customers", color="Segment",
            title="Number of Customers in Each Segment",
            labels={"Customers": "Customers", "Segment": "Customer segment"}
        )
        fig.update_layout(showlegend=False, xaxis_tickangle=-20)
        st.plotly_chart(fig, use_container_width=True)
    with right:
        fig = px.bar(
            cluster_summary.sort_values("TotalRevenue", ascending=False),
            x="Segment", y="TotalRevenue", color="Segment",
            title="Revenue Contribution by Segment",
            labels={"TotalRevenue": "Total sales (£)", "Segment": "Customer segment"}
        )
        fig.update_layout(showlegend=False, xaxis_tickangle=-20)
        st.plotly_chart(fig, use_container_width=True)

    fig = px.scatter(
        rfm, x="Frequency", y="Monetary", color="Segment",
        hover_data=["CustomerID", "Recency", "TotalItems"],
        title="Customer Map: Purchase Frequency vs Total Spend",
        labels={"Frequency": "Number of distinct orders", "Monetary": "Total spend (£)"}
    )
    st.plotly_chart(fig, use_container_width=True)

    fig = px.bar(
        cluster_summary.sort_values("Cluster"),
        x="Segment", y="AvgRecency", color="Segment",
        title="Average Recency by Segment (lower is more recent)",
        labels={"AvgRecency": "Days since last purchase", "Segment": "Customer segment"}
    )
    fig.update_layout(showlegend=False, xaxis_tickangle=-20)
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.markdown("#### Segment profile")
    display_summary = cluster_summary.copy()
    display_summary["AvgRecency"] = display_summary["AvgRecency"].round(1)
    display_summary["AvgFrequency"] = display_summary["AvgFrequency"].round(1)
    display_summary["AvgMonetary"] = display_summary["AvgMonetary"].round(2)
    display_summary["TotalRevenue"] = display_summary["TotalRevenue"].round(2)
    display_summary = display_summary.rename(columns={
        "Cluster": "Cluster ID", "Customers": "Customers",
        "AvgRecency": "Avg. recency (days)", "AvgFrequency": "Avg. orders",
        "AvgMonetary": "Avg. spend (£)", "TotalRevenue": "Segment sales (£)"
    })
    st.dataframe(display_summary, use_container_width=True, hide_index=True)

    st.markdown("#### Interpretation")
    st.write(
        "- **Champions / High Value:** high-spending customers who also purchase frequently.\n"
        "- **High Spenders:** customers with high average monetary value.\n"
        "- **Frequent Shoppers:** customers who place many distinct orders.\n"
        "- **At-Risk / Inactive:** customers whose last purchase was relatively long ago.\n"
        "- **Regular Customers:** remaining customers with more moderate behaviour.\n\n"
        "These labels are heuristic summaries of each cluster, not ground-truth categories. "
        "A label may be reused or may not appear depending on the dataset and selected K."
    )

with tab3:
    st.markdown("#### Filter and inspect customers")
    selected_segments = st.multiselect(
        "Choose segments", options=sorted(rfm["Segment"].unique()),
        default=sorted(rfm["Segment"].unique())
    )
    max_spend = float(rfm["Monetary"].quantile(1.0))
    spend_range = st.slider(
        "Total customer spend (£)",
        min_value=0.0, max_value=max(1.0, max_spend),
        value=(0.0, max(1.0, max_spend)), step=max(0.01, max_spend / 1000)
    )
    filtered = rfm[
        rfm["Segment"].isin(selected_segments)
        & rfm["Monetary"].between(spend_range[0], spend_range[1])
    ].copy()
    filtered = filtered.sort_values("Monetary", ascending=False)
    show = filtered[["CustomerID", "Segment", "Recency", "Frequency", "Monetary", "TotalItems", "LastPurchase"]].copy()
    show["Monetary"] = show["Monetary"].round(2)
    show["LastPurchase"] = pd.to_datetime(show["LastPurchase"]).dt.strftime("%Y-%m-%d")
    st.write(f"Showing **{len(show):,}** customers")
    st.dataframe(show, use_container_width=True, hide_index=True)
    st.download_button(
        "⬇️ Download customer segmentation CSV",
        data=filtered.to_csv(index=False).encode("utf-8"),
        file_name="customer_segmentation_results.csv",
        mime="text/csv"
    )

with tab4:
    st.markdown("#### Machine learning workflow")
    st.markdown("""
1. **Load data:** read the Online Retail transaction CSV.
2. **Clean data:** remove rows missing required fields, cancelled invoices, and non-positive quantities/prices.
3. **Calculate sales:** `Sales = Quantity × UnitPrice`.
4. **Build RFM features:**
   - **Recency:** days since the customer's most recent purchase.
   - **Frequency:** number of distinct invoices/orders.
   - **Monetary:** total customer spend.
5. **Transform and scale:** apply `log1p` to RFM values and standardise features so their scales are comparable.
6. **Cluster:** use K-Means with the selected K and a fixed random seed for reproducibility.
7. **Interpret:** compare cluster-level recency, frequency, and monetary values.
""")
    st.markdown("#### Model diagnostics")
    d1, d2 = st.columns(2)
    d1.metric("Selected K", n_clusters)
    d2.metric("Silhouette score", f"{score:.3f}" if pd.notna(score) else "Not available")
    st.caption("Silhouette score ranges roughly from -1 to 1; a higher value generally indicates better-separated clusters. It should be interpreted alongside business usefulness.")
    st.markdown("#### Limitations")
    st.write(
        "K-Means requires choosing K and is sensitive to feature engineering and outliers. "
        "The current implementation uses RFM features only. Customer IDs are treated as identifiers, "
        "not model features. The data snapshot date is based on the latest valid transaction in the file."
    )

st.divider()
st.caption("Machine Learning Lab Projects – 2026 | Algorithm: K-Means | Dataset: Online Retail Dataset (Kaggle)")
