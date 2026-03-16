# Transaction Behavior Clustering

Transaction Behavior Clustering is a portfolio-ready machine learning project that groups credit card customers into behavior-based segments and presents the results in an interactive Streamlit dashboard. The goal is to mirror the kind of customer analytics work used by banks and fintech teams to understand spending patterns, repayment behavior, and engagement levels.

## Project Overview

This project uses unsupervised learning to:

- load and clean customer transaction behavior data
- prepare numeric features for modeling
- evaluate multiple cluster counts using silhouette score
- fit a final KMeans segmentation model
- reduce the feature space with PCA for visualization
- generate business-friendly segment names and recommendations
- export customer- and cluster-level reports
- present everything in a Streamlit dashboard

## Dataset

The project uses a credit card customer behavior dataset stored locally at `data/ccdata.csv`. It contains account-level variables related to balances, purchases, cash advances, payment behavior, credit limits, and tenure. Identifier fields are removed from the modeling pipeline so clustering is driven by customer financial behavior rather than customer IDs.

## Machine Learning Pipeline

### 1. Preprocessing

The preprocessing step:

- loads the CSV file
- removes identifier columns such as `CUST_ID`
- keeps numeric features for clustering
- fills missing values with feature medians
- scales features using `StandardScaler`

### 2. Clustering

The clustering pipeline:

- tests `k = 2` through `k = 8`
- calculates silhouette score for each option
- automatically selects the best `k`
- trains a final `KMeans` model
- assigns a cluster label to every customer

### 3. Interpretation

Cluster summaries are converted into business-ready segment profiles. The project generates:

- average feature values by cluster
- readable segment names
- short recommendations that connect each segment to possible banking actions

### 4. Visualization

PCA reduces the scaled feature matrix to two components so clusters can be explored visually in the dashboard.

## Dashboard Features

The Streamlit app includes:

- dataset overview metrics
- silhouette score evaluation chart
- PCA scatter plot of customer clusters
- cluster distribution bar chart
- segment profile table
- business insight section for each cluster
- direct CSV downloads for generated reports

## Outputs

Running the app creates the following files in `reports/`:

- `customer_segments.csv`: original customer records with assigned cluster labels and PCA coordinates
- `cluster_summary.csv`: cluster-level averages, segment names, and recommended actions

## Business Use Cases

This type of segmentation can support:

- targeted credit card marketing campaigns
- customer retention strategies
- risk monitoring for revolving balances
- product design for premium users
- reactivation campaigns for low-engagement customers

## Project Structure

```text
Girijaesh-Transaction-Behavior-Clustering/
├── app.py
├── data/
│   └── ccdata.csv
├── reports/
│   ├── customer_segments.csv
│   └── cluster_summary.csv
├── src/
│   ├── cluster.py
│   ├── explain.py
│   ├── preprocess.py
│   └── utils.py
├── requirements.txt
├── README.md
└── .gitignore
```

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Start the dashboard:

```bash
streamlit run app.py
```

Then open the local Streamlit URL shown in the terminal.

## Why This Project Works Well in a Portfolio

This repository shows more than just a model notebook. It demonstrates modular project structure, automated model selection, business-focused interpretation, report generation, and a lightweight analytics application that someone can actually run and explore.
