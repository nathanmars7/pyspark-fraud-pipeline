# 🏦 PySpark Fraud Pipeline

A end-to-end payment fraud detection pipeline built with **PySpark** and **Delta Lake** on **Databricks**.

This project simulates a real-world data engineering pipeline at a fintech company — generating synthetic transaction data, engineering fraud signals, building a lakehouse architecture, and adding production-grade data quality monitoring.

---

## 🏗️ Architecture

```
Bronze Layer          Silver Layer            Gold Layer               Monitoring
─────────────         ────────────            ──────────               ──────────
fraud_transactions --> fraud_features    -->  fraud_user_profile       fraud_dq_report
(raw transactions)    (window features)  -->  fraud_merchant_stats
                                         -->  fraud_gold_enriched
```

This follows the **Medallion Architecture** — the industry-standard pattern for data lakehouses used at companies like Databricks, Revolut, and Stripe.

---

## 📓 Notebooks

| # | Notebook | Key Concepts |
|---|---|---|
| 01 | `nb01_payment_fraud_setup_and_python_refresher` | Python lists, dicts, list comprehensions, lambdas, pandas, PySpark orientation |
| 02 | `nb02_synthetic_data_and_delta_table` | Synthetic data generation, Delta Lake, `saveAsTable()`, time travel |
| 03 | `nb03_window_functions_and_feature_engineering` | Window functions, velocity features, behavioural features, risk scoring |
| 04 | `nb04_aggregations_joins_and_gold_layer` | Gold layer tables, left/inner/anti joins, reusable transformation functions |
| 05 | `nb05_data_quality_and_monitoring` | DataQualityChecker class, drift detection, Delta constraints, DQ report table |

---

## 🔍 What the Pipeline Does

### 1. Data Generation (Notebook 2)
Generates 10,000 synthetic payment transactions with realistic fraud patterns:
- High-risk countries (Nigeria, Russia, Brazil) have elevated fraud rates
- Crypto payments have the highest fraud rate of all payment methods
- Fraudulent transactions skew toward late-night hours and larger amounts
- ~5% overall fraud rate, matching real-world benchmarks

### 2. Feature Engineering (Notebook 3)
Builds fraud detection signals using **PySpark Window functions**:
- **Velocity features** — transaction count in last 1h, 24h, 7 days
- **Spend velocity** — total amount spent in rolling time windows
- **Behavioural features** — is this amount unusual for this specific user?
- **Amount ratio** — transaction amount vs user's historical average
- **Composite risk score** — combines 7 signals into a `LOW / MEDIUM / HIGH` label

### 3. Gold Layer (Notebook 4)
Builds aggregated dimension tables for analysts and ML models:
- **User profile table** — spend profile, fraud history, risk tier per user
- **Merchant stats table** — fraud rates by merchant and category
- **Enriched Gold table** — fully joined transactions with user and merchant context
- Reusable transformation functions using PySpark's `.transform()` pattern

### 4. Data Quality & Monitoring (Notebook 5)
Production-grade pipeline guardrails:
- `DataQualityChecker` class — chainable checks for nulls, ranges, uniqueness, allowed values
- **Fraud rate drift detection** — flags months where fraud rate deviates >50% from baseline
- **Delta constraints** — rejects bad data at write time (`amount > 0`, `is_fraud IN (0,1)`)
- Full DQ report saved as a Delta table for trending over time

---

## 🛠️ Tech Stack

| Tool | Version | Purpose |
|---|---|---|
| Apache Spark | 3.x | Distributed data processing |
| PySpark | 3.x | Python API for Spark |
| Delta Lake | 3.x | ACID transactions, time travel, schema enforcement |
| Databricks | Free Edition | Managed Spark environment |
| Python | 3.x | Data generation, helper functions |
| pandas | 2.x | Small-data exploration and seeding |

---

## 🚀 How to Run

### Prerequisites
- A [Databricks](https://www.databricks.com/) account (Free Edition works)
- No additional dependencies — all libraries are pre-installed on Databricks

### Setup
1. Clone this repository
2. In your Databricks workspace, create a folder called `pyspark-fraud-pipeline`
3. Import each `.py` notebook file into the folder (File → Import → File)
4. Run notebooks in order: `nb01` → `nb02` → `nb03` → `nb04` → `nb05`

### Important
Each notebook starts with:
```python
spark.sql("USE CATALOG workspace")
spark.sql("USE SCHEMA default")
```
This pins the catalog and schema so all tables resolve correctly.

---

## 📊 Key Results

After running the full pipeline:

| Metric | Value |
|---|---|
| Total transactions | 10,000 |
| Fraud rate | ~5% |
| Features engineered | 15+ |
| Delta tables created | 6 |
| DQ checks run | 30+ |
| Pipeline layers | Bronze → Silver → Gold |

**Risk score performance:**
- `HIGH` risk label captures a significantly elevated fraud rate vs the 5% baseline
- `LOW` risk label captures the majority of legitimate transactions
- Combined user + merchant risk tier further improves signal quality

---

## 🧠 Concepts Covered

- **PySpark DataFrames** — `filter()`, `select()`, `withColumn()`, `groupBy()`, `orderBy()`
- **Window Functions** — `partitionBy()`, `orderBy()`, `rangeBetween()`, `rowsBetween()`
- **Ranking Functions** — `row_number()`, `rank()`, `dense_rank()`
- **Join Types** — inner, left, anti — when to use each
- **Delta Lake** — ACID writes, time travel, `ALTER TABLE ... ADD CONSTRAINT`
- **Medallion Architecture** — Bronze / Silver / Gold layer pattern
- **Data Quality** — null checks, range checks, uniqueness, drift detection
- **Production Patterns** — reusable transformation functions, `.transform()` chaining

---

## 📁 Repository Structure

```
pyspark-fraud-pipeline/
│
├── README.md
│
├── nb01_payment_fraud_setup_and_python_refresher.py
├── nb02_synthetic_data_and_delta_table.py
├── nb03_window_functions_and_feature_engineering.py
├── nb04_aggregations_joins_and_gold_layer.py
└── nb05_data_quality_and_monitoring.py
```

---

## ⚠️ Known Limitations

- **Synthetic data** — fraud patterns are simulated, not learned from real transactions
- **No ML model** — the risk score is rule-based; a production system would use a trained classifier
- **Single table grain** — `fraud_merchant_stats` uses `(merchant_id, merchant_category)` as its natural key; the uniqueness check on `merchant_id` alone will flag duplicates by design
- **Community Edition** — public DBFS root is disabled; all tables use `workspace.default` as the catalog and schema

---

## 🔜 What's Next

- [ ] Notebook 6 — ML fraud classifier using Spark MLlib or MLflow
- [ ] Orchestration — schedule the pipeline with Databricks Workflows
- [ ] Streaming — replace batch ingestion with Spark Structured Streaming

---

## 👤 Author

**Nathan**
GitHub: [@nathanmars7](https://github.com/nathanmars7)

---

## 📄 Licence

MIT — free to use, adapt, and share.
