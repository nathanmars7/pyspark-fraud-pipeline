# Databricks notebook source
# MAGIC %md
# MAGIC # 🏦 Payment Fraud Pipeline — Notebook 2
# MAGIC ## Synthetic Data Generation + Your First Delta Table
# MAGIC
# MAGIC **Project:** Payment Fraud Detection Pipeline
# MAGIC **Series:** Notebook 2 of N
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 🎯 What you'll learn in this notebook
# MAGIC
# MAGIC 1. Generate a realistic 10,000-row synthetic payment fraud dataset using Python
# MAGIC 2. Understand **Delta Lake** — what it is and why Databricks uses it
# MAGIC 3. Save your dataset as a **Delta table** (your first proper data lake table)
# MAGIC 4. Load it back with PySpark and run your first real exploratory analysis
# MAGIC 5. New PySpark operations: `withColumn()`, `groupBy()`, `orderBy()`, `alias()`
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 🧠 Why generate synthetic data?
# MAGIC
# MAGIC Real fraud datasets are either:
# MAGIC - Locked behind NDAs at financial companies
# MAGIC - Heavily anonymised and small when public
# MAGIC - Not designed for learning pipeline concepts
# MAGIC
# MAGIC By generating our own, we control:
# MAGIC - The schema (exactly the fields we want)
# MAGIC - The fraud rate and patterns (so we know the "ground truth")
# MAGIC - The volume (we can scale up or down)
# MAGIC
# MAGIC This is also a real skill — data engineers often generate synthetic data
# MAGIC for testing pipelines before production data is available.

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Setup — Catalog and Schema
# MAGIC
# MAGIC We set the catalog and schema explicitly at the top of every notebook.
# MAGIC This means we can refer to tables by short name (`fraud_transactions`)
# MAGIC throughout the notebook without needing the full three-part path.

# COMMAND ----------

# Pin the catalog and schema for this entire notebook session
spark.sql("USE CATALOG workspace")
spark.sql("USE SCHEMA default")

# Confirm where we are
spark.sql("SELECT current_catalog(), current_schema()").show()

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Part 1 — Generating the Synthetic Dataset
# MAGIC ### 1.1 The Plan
# MAGIC
# MAGIC We'll simulate 10,000 payment transactions with realistic fraud patterns baked in:
# MAGIC
# MAGIC | Fraud Signal | Logic |
# MAGIC |---|---|
# MAGIC | High-risk countries | Transactions from certain countries have higher fraud rates |
# MAGIC | Large amounts | Very high amounts are more likely to be fraudulent |
# MAGIC | Certain payment methods | Some methods are riskier than others |
# MAGIC | Night-time transactions | Late night hours have elevated fraud rates |
# MAGIC | New users | Users with low transaction history are higher risk |
# MAGIC
# MAGIC These patterns mirror what real fraud detection systems look for.
# MAGIC We're building the dataset so that our later analysis can actually find these signals.

# COMMAND ----------

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# Set a random seed so results are reproducible
# This means every time you run this notebook, you get the same dataset
np.random.seed(42)
random.seed(42)

print("Libraries imported. Ready to generate data.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.2 Define the Building Blocks
# MAGIC
# MAGIC First we define the "universe" of values our transactions can draw from.
# MAGIC We assign fraud weights to countries and payment methods —
# MAGIC higher weight = more likely to appear in fraudulent transactions.

# COMMAND ----------

# Countries with their fraud likelihood weights
# Higher fraud weight = more likely to appear in fraudulent transactions
COUNTRIES = {
    "GB": {"legit": 25, "fraud": 10},
    "US": {"legit": 25, "fraud": 10},
    "JP": {"legit": 15, "fraud":  5},
    "DE": {"legit": 10, "fraud":  5},
    "FR": {"legit": 10, "fraud":  5},
    "NG": {"legit":  3, "fraud": 20},  # high fraud weight
    "RU": {"legit":  3, "fraud": 20},  # high fraud weight
    "BR": {"legit":  5, "fraud": 15},  # elevated fraud weight
    "CN": {"legit":  3, "fraud": 10},
    "IN": {"legit":  1, "fraud":  0},
}

# Payment methods with fraud weights
PAYMENT_METHODS = {
    "card":          {"legit": 50, "fraud": 30},
    "bank_transfer": {"legit": 25, "fraud": 15},
    "wallet":        {"legit": 15, "fraud": 20},
    "crypto":        {"legit":  5, "fraud": 30},  # high fraud weight
    "voucher":       {"legit":  5, "fraud":  5},
}

# Merchant categories
MERCHANT_CATEGORIES = [
    "electronics", "groceries", "travel", "entertainment",
    "clothing", "restaurant", "utilities", "gaming", "luxury"
]

print("Configuration defined.")
print(f"Countries: {list(COUNTRIES.keys())}")
print(f"Payment methods: {list(PAYMENT_METHODS.keys())}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.3 Helper Functions
# MAGIC
# MAGIC We'll write small, focused helper functions for each part of the data generation.
# MAGIC This is good Python practice — each function does one thing clearly.
# MAGIC
# MAGIC Note the use of `np.random.choice()` with a `p` (probability) parameter.
# MAGIC This lets us control how often each value appears.

# COMMAND ----------

def weighted_choice(options_dict, key):
    """
    Pick a random item from a dict where each item has a weight for 'legit' or 'fraud'.
    key: either 'legit' or 'fraud'
    """
    items = list(options_dict.keys())
    weights = [options_dict[item][key] for item in items]
    total = sum(weights)
    probs = [w / total for w in weights]
    return np.random.choice(items, p=probs)


def generate_amount(is_fraud):
    """
    Generate a transaction amount.
    Fraudulent transactions skew toward higher amounts.
    """
    if is_fraud:
        if random.random() < 0.3:
            return round(random.uniform(2000, 15000), 2)  # large fraudulent amount
        else:
            return round(random.uniform(50, 2000), 2)     # mid-range (harder to detect)
    else:
        if random.random() < 0.05:
            return round(random.uniform(500, 5000), 2)    # genuine large purchase
        else:
            return round(random.uniform(1, 500), 2)       # typical everyday transaction


def generate_timestamp(is_fraud, start_date, end_date):
    """
    Generate a transaction timestamp.
    Fraudulent transactions are more common at night (0:00-5:00).
    """
    delta = end_date - start_date
    random_day = start_date + timedelta(days=random.randint(0, delta.days))

    if is_fraud and random.random() < 0.4:
        hour = random.randint(0, 4)   # 40% of fraud happens midnight to 5am
    else:
        hour = int(np.clip(np.random.normal(13, 4), 6, 23))  # business hours

    minute = random.randint(0, 59)
    second = random.randint(0, 59)
    return random_day.replace(hour=hour, minute=minute, second=second)


print("Helper functions defined.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.4 Generate the Dataset
# MAGIC
# MAGIC Now we put it all together. We'll generate 10,000 transactions with a ~5% fraud rate
# MAGIC (realistic — real fraud rates are typically 1-10%).

# COMMAND ----------

N_TRANSACTIONS = 10_000
FRAUD_RATE = 0.05
N_USERS = 500
N_MERCHANTS = 200

start_date = datetime(2024, 1, 1)
end_date = datetime(2024, 12, 31)

rows = []

for i in range(N_TRANSACTIONS):
    is_fraud = 1 if random.random() < FRAUD_RATE else 0
    txn_type = "fraud" if is_fraud else "legit"

    row = {
        "transaction_id":    f"txn_{i+1:06d}",
        "user_id":           f"usr_{random.randint(1, N_USERS):04d}",
        "merchant_id":       f"mrc_{random.randint(1, N_MERCHANTS):04d}",
        "merchant_category": random.choice(MERCHANT_CATEGORIES),
        "amount":            generate_amount(is_fraud),
        "country":           weighted_choice(COUNTRIES, txn_type),
        "payment_method":    weighted_choice(PAYMENT_METHODS, txn_type),
        "timestamp":         generate_timestamp(is_fraud, start_date, end_date),
        "is_fraud":          is_fraud,
    }
    rows.append(row)

df_raw = pd.DataFrame(rows)
df_raw = df_raw.sort_values("timestamp").reset_index(drop=True)

print(f"Generated {len(df_raw):,} transactions")
print(f"Fraud count: {df_raw['is_fraud'].sum():,} ({df_raw['is_fraud'].mean()*100:.1f}%)")
print(f"Date range: {df_raw['timestamp'].min()} -> {df_raw['timestamp'].max()}")
df_raw.head()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.5 Sanity Checks
# MAGIC
# MAGIC Before saving, we verify the fraud patterns we built in are actually showing up.
# MAGIC This is a habit worth keeping in every pipeline — always validate before writing.

# COMMAND ----------

# Country distribution — high-risk countries (NG, RU, BR) should show more fraud
print("=== Country distribution (fraud vs legit) ===")
print(df_raw.groupby(["country", "is_fraud"]).size().unstack(fill_value=0))

# COMMAND ----------

# Amount distribution — fraud should skew higher
print("=== Average amount: fraud vs legit ===")
print(df_raw.groupby("is_fraud")["amount"].describe().round(2))

# COMMAND ----------

# Payment method fraud rates — crypto should be highest
print("=== Payment method fraud breakdown ===")
pm_stats = df_raw.groupby("payment_method").agg(
    total=("is_fraud", "count"),
    fraud=("is_fraud", "sum")
)
pm_stats["fraud_rate"] = (pm_stats["fraud"] / pm_stats["total"] * 100).round(1)
print(pm_stats.sort_values("fraud_rate", ascending=False))

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Part 2 — Delta Lake
# MAGIC ### 2.1 What is Delta Lake?
# MAGIC
# MAGIC When you save data in Databricks, you don't just save a CSV or Parquet file.
# MAGIC You save it as a **Delta table** — and that's a meaningful upgrade.
# MAGIC
# MAGIC | Feature | Plain CSV/Parquet | Delta Lake |
# MAGIC |---|---|---|
# MAGIC | ACID transactions | ❌ | ✅ |
# MAGIC | Schema enforcement | ❌ | ✅ |
# MAGIC | Time travel (see old versions) | ❌ | ✅ |
# MAGIC | Efficient upserts (MERGE) | ❌ | ✅ |
# MAGIC | Audit log | ❌ | ✅ |
# MAGIC
# MAGIC **ACID** means: if a write fails halfway through, the table is not corrupted.
# MAGIC
# MAGIC **Time travel** means: you can query the table as it looked at any previous version.
# MAGIC Extremely useful for debugging fraud pipelines.
# MAGIC
# MAGIC In Databricks, Delta is the default format. You get all of this for free.

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.2 Convert to Spark and Save as Delta Table
# MAGIC
# MAGIC We use `saveAsTable()` with the full three-part name `catalog.schema.table`.
# MAGIC This does two things at once:
# MAGIC 1. Writes the Delta files to managed storage
# MAGIC 2. Registers the table name in the metastore
# MAGIC
# MAGIC Because we set `USE CATALOG workspace` and `USE SCHEMA default` at the top,
# MAGIC we could also just write `"fraud_transactions"` — but the full path is explicit
# MAGIC and clearer for a portfolio project.

# COMMAND ----------

from pyspark.sql import functions as F

# Convert pandas -> Spark
df_spark = spark.createDataFrame(df_raw)

# Always check schema after conversion
df_spark.printSchema()

# COMMAND ----------

# Drop the table first if it already exists, then write clean
spark.sql("DROP TABLE IF EXISTS workspace.default.fraud_transactions")

df_spark.write \
    .format("delta") \
    .mode("overwrite") \
    .saveAsTable("workspace.default.fraud_transactions")

print(f"✅ Delta table written: workspace.default.fraud_transactions")
print(f"   Rows written: {df_spark.count():,}")

# COMMAND ----------

# Verify it's registered
spark.sql("SHOW TABLES IN workspace.default").show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.3 Managed Tables — What Just Happened
# MAGIC
# MAGIC When you called `saveAsTable("workspace.default.fraud_transactions")`, Spark did two things:
# MAGIC
# MAGIC 1. **Wrote the Delta files** to a location Databricks manages for you
# MAGIC 2. **Registered the table name** in the metastore — queryable by name from any notebook
# MAGIC
# MAGIC | | Managed Table | External Table |
# MAGIC |---|---|---|
# MAGIC | Storage location | Databricks controls it | You specify the path |
# MAGIC | Drop table = delete data? | ✅ Yes | ❌ No, just removes the pointer |
# MAGIC | Best for | Learning, dev work | Production pipelines |
# MAGIC
# MAGIC For this project, managed tables are perfect.

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.4 Delta Time Travel
# MAGIC
# MAGIC Every write to a Delta table creates a new version automatically.
# MAGIC You can query any previous version — this is called **time travel**.
# MAGIC It's free and requires no setup.

# COMMAND ----------

# See the full version history of our table
spark.sql("DESCRIBE HISTORY workspace.default.fraud_transactions").select(
    "version", "timestamp", "operation", "operationMetrics"
).show(truncate=False)

# COMMAND ----------

# Read version 0 (the original write) — time travel in action
df_v0 = spark.read \
    .format("delta") \
    .option("versionAsOf", 0) \
    .table("workspace.default.fraud_transactions")

print(f"Version 0 row count: {df_v0.count():,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Part 3 — Loading and Exploring with PySpark
# MAGIC ### 3.1 Load the Delta Table
# MAGIC
# MAGIC Now let's pretend we're a new engineer opening this project for the first time.
# MAGIC We load the table fresh and explore it — this is your standard pipeline start.

# COMMAND ----------

# Load from the registered table name
df = spark.table("workspace.default.fraud_transactions")

df.printSchema()

# COMMAND ----------

print(f"Total rows: {df.count():,}")
print(f"Columns: {df.columns}")

# COMMAND ----------

df.show(5, truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.2 withColumn() — Adding Derived Columns
# MAGIC
# MAGIC `withColumn(name, expression)` either adds a new column or replaces an existing one.
# MAGIC You chain multiple calls to build up transformations step by step.
# MAGIC
# MAGIC Here we extract useful time features from the timestamp —
# MAGIC these will become fraud signals in Notebook 3.

# COMMAND ----------

df_enriched = df \
    .withColumn("hour_of_day",   F.hour("timestamp")) \
    .withColumn("day_of_week",   F.dayofweek("timestamp")) \
    .withColumn("month",         F.month("timestamp")) \
    .withColumn("is_night",      (F.hour("timestamp") <= 5).cast("int")) \
    .withColumn("is_high_value", (F.col("amount") > 1000).cast("int"))

df_enriched.select(
    "transaction_id", "timestamp", "amount",
    "hour_of_day", "day_of_week", "is_night", "is_high_value"
).show(10, truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.3 groupBy() and Aggregations
# MAGIC
# MAGIC **Pattern:**
# MAGIC ```python
# MAGIC df.groupBy("column").agg(
# MAGIC     F.count("*").alias("total"),
# MAGIC     F.sum("amount").alias("total_amount")
# MAGIC )
# MAGIC ```
# MAGIC Always use `.alias()` to name your output columns clearly.

# COMMAND ----------

# Fraud rate by country
df_enriched.groupBy("country").agg(
    F.count("*").alias("total_transactions"),
    F.sum("is_fraud").alias("fraud_count"),
    F.round(F.avg("amount"), 2).alias("avg_amount"),
    F.round(F.sum("is_fraud") / F.count("*") * 100, 2).alias("fraud_rate_pct")
).orderBy(F.col("fraud_rate_pct").desc()).show(truncate=False)

# COMMAND ----------

# Fraud rate by payment method
df_enriched.groupBy("payment_method").agg(
    F.count("*").alias("total_transactions"),
    F.sum("is_fraud").alias("fraud_count"),
    F.round(F.sum("is_fraud") / F.count("*") * 100, 2).alias("fraud_rate_pct"),
    F.round(F.avg("amount"), 2).alias("avg_amount")
).orderBy(F.col("fraud_rate_pct").desc()).show(truncate=False)

# COMMAND ----------

# Night-time vs daytime fraud comparison
df_enriched.groupBy("is_night").agg(
    F.count("*").alias("total_transactions"),
    F.sum("is_fraud").alias("fraud_count"),
    F.round(F.sum("is_fraud") / F.count("*") * 100, 2).alias("fraud_rate_pct")
).orderBy("is_night").show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.4 Querying with Pure SQL
# MAGIC
# MAGIC One of the best things about a registered table is you can switch to
# MAGIC pure SQL using `%sql` in any cell — no Python needed.
# MAGIC This is exactly how analysts and engineers work side by side in Databricks.

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC   country,
# MAGIC   COUNT(*) AS total_transactions,
# MAGIC   SUM(is_fraud) AS fraud_count,
# MAGIC   ROUND(SUM(is_fraud) / COUNT(*) * 100, 2) AS fraud_rate_pct
# MAGIC FROM workspace.default.fraud_transactions
# MAGIC GROUP BY country
# MAGIC ORDER BY fraud_rate_pct DESC

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC   payment_method,
# MAGIC   ROUND(AVG(amount), 2) AS avg_amount,
# MAGIC   SUM(is_fraud) AS fraud_count,
# MAGIC   COUNT(*) AS total
# MAGIC FROM workspace.default.fraud_transactions
# MAGIC GROUP BY payment_method
# MAGIC ORDER BY fraud_count DESC

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.5 Top Fraudulent Transactions
# MAGIC
# MAGIC Combining `.filter()`, `.select()`, and `.orderBy()` in a chain —
# MAGIC a pattern you'll write constantly in real pipelines.

# COMMAND ----------

# Top 10 highest-value fraud transactions
df_enriched \
    .filter(F.col("is_fraud") == 1) \
    .select("transaction_id", "user_id", "amount", "country", "payment_method", "hour_of_day") \
    .orderBy(F.col("amount").desc()) \
    .limit(10) \
    .show(truncate=False)

# COMMAND ----------

# Users with the most fraud transactions (potential compromised accounts)
df_enriched \
    .filter(F.col("is_fraud") == 1) \
    .groupBy("user_id") \
    .agg(
        F.count("*").alias("fraud_transactions"),
        F.round(F.sum("amount"), 2).alias("total_fraud_amount")
    ) \
    .orderBy(F.col("fraud_transactions").desc()) \
    .limit(10) \
    .show(truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## ✅ Notebook 2 Complete
# MAGIC
# MAGIC ### What you covered:
# MAGIC
# MAGIC | Topic | Key Takeaway |
# MAGIC |---|---|
# MAGIC | Catalog + schema setup | Always `USE CATALOG` and `USE SCHEMA` at the top of every notebook |
# MAGIC | Synthetic data generation | Control schema, fraud rate, and patterns — a real engineering skill |
# MAGIC | `np.random.choice` with weights | Drive realistic distributions in test data |
# MAGIC | Delta Lake | ACID, schema enforcement, time travel — the default Databricks format |
# MAGIC | `saveAsTable()` | Writes Delta files AND registers the table name in one call |
# MAGIC | `withColumn()` | Add/replace columns — your primary transformation tool |
# MAGIC | `groupBy().agg()` | "For each X, calculate Y" in PySpark |
# MAGIC | `orderBy()` | Sort results — always explicit `.desc()` or `.asc()` |
# MAGIC | `%sql` cells | Switch to pure SQL anytime — same table, different interface |
# MAGIC | Delta time travel | Query historical versions for free |
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### What's next — Notebook 3:
# MAGIC
# MAGIC - **Window functions** — the most powerful concept in fraud detection
# MAGIC - Rolling transaction counts per user (velocity features)
# MAGIC - User's average spend over the last 7 days
# MAGIC - Flag transactions that are 3x higher than a user's typical amount
# MAGIC - Build a composite **risk score** column