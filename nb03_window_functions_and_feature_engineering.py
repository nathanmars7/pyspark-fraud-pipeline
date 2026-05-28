# Databricks notebook source
# MAGIC %md
# MAGIC # 🏦 Payment Fraud Pipeline — Notebook 3
# MAGIC ## Window Functions & Feature Engineering
# MAGIC
# MAGIC **Project:** Payment Fraud Detection Pipeline
# MAGIC **Series:** Notebook 3 of N
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 🎯 What you'll learn in this notebook
# MAGIC
# MAGIC 1. What window functions are and why they are the core of fraud detection
# MAGIC 2. `Window.partitionBy()` and `Window.orderBy()` — the two building blocks
# MAGIC 3. Velocity features: how many transactions has a user made in the last hour?
# MAGIC 4. Behavioural features: is this amount unusual for this user?
# MAGIC 5. Ranking functions: `row_number()`, `rank()`, `dense_rank()`
# MAGIC 6. Build a composite **risk score** by combining multiple signals
# MAGIC 7. Save the enriched feature table as a new Delta table
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 🧠 Why window functions for fraud detection?
# MAGIC
# MAGIC Most fraud signals are not about a single transaction in isolation.
# MAGIC They are about **patterns over time** for a specific user or account:
# MAGIC
# MAGIC - A user who normally spends £20/day suddenly spends £4,000 → suspicious
# MAGIC - A card that makes 10 transactions in 5 minutes → suspicious
# MAGIC - A user's first-ever transaction is for £8,000 → suspicious
# MAGIC
# MAGIC To calculate these patterns, you need to look at a user's **history**
# MAGIC while processing each row. That is exactly what window functions do.
# MAGIC
# MAGIC A window function computes a value for each row **based on a group of
# MAGIC related rows** — without collapsing the DataFrame the way `groupBy()` does.

# COMMAND ----------

# Set catalog and schema at the top of every notebook
spark.sql("USE CATALOG workspace")
spark.sql("USE SCHEMA default")

spark.sql("SELECT current_catalog(), current_schema()").show()

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.window import Window

# Load our fraud transactions table
df = spark.table("workspace.default.fraud_transactions")

print(f"Loaded {df.count():,} rows")
df.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Part 1 — Understanding Window Functions
# MAGIC ### 1.1 The Core Concept
# MAGIC
# MAGIC A window function requires two things:
# MAGIC
# MAGIC 1. **`partitionBy()`** — defines the "group" for each row.
# MAGIC    Think of it as: "do this calculation separately for each user"
# MAGIC    This is like the `GROUP BY` key — but the rows are NOT collapsed.
# MAGIC
# MAGIC 2. **`orderBy()`** — defines the order within each partition.
# MAGIC    Usually a timestamp — "look at events in chronological order"
# MAGIC
# MAGIC **The key difference from groupBy():**
# MAGIC
# MAGIC ```
# MAGIC groupBy():  1000 rows → 10 rows  (collapses — one row per group)
# MAGIC window:     1000 rows → 1000 rows (preserves — adds a column to each row)
# MAGIC ```
# MAGIC
# MAGIC This is critical. In fraud detection, we want to keep every transaction
# MAGIC and enrich it with context about that user's history.

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.2 Your First Window — Running Transaction Count Per User
# MAGIC
# MAGIC Let's start simple: for each transaction, how many transactions has
# MAGIC this user made in total up to and including this one?
# MAGIC
# MAGIC This is called a **cumulative count** — it runs forwards in time.

# COMMAND ----------

# Define a window: partition by user, order by time
user_window = Window \
    .partitionBy("user_id") \
    .orderBy("timestamp")

# Add a cumulative transaction count per user
df_with_count = df.withColumn(
    "user_txn_count",
    F.count("*").over(user_window)
)

# Look at one user to see how it works
df_with_count \
    .filter(F.col("user_id") == "usr_0001") \
    .select("transaction_id", "user_id", "timestamp", "amount", "user_txn_count") \
    .orderBy("timestamp") \
    .show(10, truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC Notice: each row keeps its own identity, but now has a count of how many
# MAGIC transactions this user has made so far (including the current one).
# MAGIC
# MAGIC The first transaction a user ever makes gets `user_txn_count = 1`.
# MAGIC This is a useful fraud signal — **first-time users are higher risk**.

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Part 2 — Velocity Features
# MAGIC ### 2.1 What is Velocity?
# MAGIC
# MAGIC **Velocity** = how fast is this user transacting?
# MAGIC
# MAGIC Fraudsters typically:
# MAGIC - Make many small transactions quickly to test a stolen card
# MAGIC - Then make one large transaction before the card is blocked
# MAGIC
# MAGIC To detect this, we calculate:
# MAGIC - How many transactions in the last 1 hour?
# MAGIC - How many transactions in the last 24 hours?
# MAGIC - Total amount spent in the last 24 hours?
# MAGIC
# MAGIC This requires a **range-based window** — instead of looking at all
# MAGIC previous rows, we look only within a time window.

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.2 Range-Based Windows
# MAGIC
# MAGIC To define a time-based window, we need to work in seconds.
# MAGIC PySpark's `rangeBetween()` works on numeric values, so we convert
# MAGIC the timestamp to a Unix timestamp (seconds since 1970-01-01).
# MAGIC
# MAGIC **Pattern:**
# MAGIC ```python
# MAGIC Window.partitionBy("user_id") \
# MAGIC       .orderBy("timestamp_seconds") \
# MAGIC       .rangeBetween(-3600, 0)   # -3600 seconds = last 1 hour, 0 = current row
# MAGIC ```

# COMMAND ----------

# Add a Unix timestamp column (seconds) — needed for range windows
df = df.withColumn("ts_seconds", F.unix_timestamp("timestamp"))

# Define range-based windows
# rangeBetween(-N, 0) means: from N seconds ago up to and including the current row

window_1h = Window \
    .partitionBy("user_id") \
    .orderBy("ts_seconds") \
    .rangeBetween(-3600, 0)           # last 1 hour

window_24h = Window \
    .partitionBy("user_id") \
    .orderBy("ts_seconds") \
    .rangeBetween(-86400, 0)          # last 24 hours (86400 = 24 * 60 * 60)

window_7d = Window \
    .partitionBy("user_id") \
    .orderBy("ts_seconds") \
    .rangeBetween(-604800, 0)         # last 7 days (604800 = 7 * 24 * 60 * 60)

print("Window specs defined.")

# COMMAND ----------

# Build velocity features
df_velocity = df \
    .withColumn("txn_count_1h",     F.count("*").over(window_1h)) \
    .withColumn("txn_count_24h",    F.count("*").over(window_24h)) \
    .withColumn("amount_sum_24h",   F.round(F.sum("amount").over(window_24h), 2)) \
    .withColumn("amount_sum_7d",    F.round(F.sum("amount").over(window_7d), 2)) \
    .withColumn("txn_count_7d",     F.count("*").over(window_7d))

# Preview the velocity features
df_velocity.select(
    "transaction_id", "user_id", "timestamp", "amount",
    "txn_count_1h", "txn_count_24h", "amount_sum_24h", "is_fraud"
).orderBy("user_id", "timestamp").show(15, truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.3 Do Velocity Features Separate Fraud from Legit?
# MAGIC
# MAGIC Let's check — if our features are useful, fraud transactions should
# MAGIC show higher velocity on average than legitimate ones.

# COMMAND ----------

# Compare average velocity: fraud vs legit
df_velocity.groupBy("is_fraud").agg(
    F.round(F.avg("txn_count_1h"),   2).alias("avg_txn_count_1h"),
    F.round(F.avg("txn_count_24h"),  2).alias("avg_txn_count_24h"),
    F.round(F.avg("amount_sum_24h"), 2).alias("avg_amount_sum_24h"),
    F.round(F.avg("amount_sum_7d"),  2).alias("avg_amount_sum_7d")
).orderBy("is_fraud").show()

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Part 3 — Behavioural Features
# MAGIC ### 3.1 Is This Amount Unusual for This User?
# MAGIC
# MAGIC Velocity tells us *how fast* a user is transacting.
# MAGIC Behavioural features tell us *how unusual* a transaction is
# MAGIC **compared to that user's own history**.
# MAGIC
# MAGIC Key features:
# MAGIC - **User's average spend** over the last 7 days
# MAGIC - **Amount ratio**: this transaction / user's average (is it 5x their normal?)
# MAGIC - **User's transaction number**: is this their very first transaction?

# COMMAND ----------

# Unbounded window — all of the user's history up to the current row
# "rows between unbounded preceding and current row"
user_history_window = Window \
    .partitionBy("user_id") \
    .orderBy("ts_seconds") \
    .rowsBetween(Window.unboundedPreceding, Window.currentRow)

# User's running average spend and transaction number
df_behavioural = df_velocity \
    .withColumn(
        "user_avg_amount_historical",
        F.round(F.avg("amount").over(user_history_window), 2)
    ) \
    .withColumn(
        "user_txn_number",
        F.row_number().over(
            Window.partitionBy("user_id").orderBy("timestamp")
        )
    ) \
    .withColumn(
        "is_first_txn",
        (F.col("user_txn_number") == 1).cast("int")
    ) \
    .withColumn(
        # How many times larger is this transaction vs the user's historical average?
        # A ratio of 5.0 means this transaction is 5x their normal spend
        "amount_vs_avg_ratio",
        F.round(
            F.col("amount") / F.when(
                F.col("user_avg_amount_historical") > 0,
                F.col("user_avg_amount_historical")
            ).otherwise(1.0),
            2
        )
    )

# Preview behavioural features
df_behavioural.select(
    "transaction_id", "user_id", "amount",
    "user_avg_amount_historical", "amount_vs_avg_ratio",
    "user_txn_number", "is_first_txn", "is_fraud"
).orderBy("user_id", "timestamp").show(15, truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.2 Do Behavioural Features Separate Fraud?

# COMMAND ----------

# Fraud vs legit on behavioural features
df_behavioural.groupBy("is_fraud").agg(
    F.round(F.avg("amount_vs_avg_ratio"), 2).alias("avg_amount_ratio"),
    F.round(F.avg("user_txn_number"),     2).alias("avg_txn_number"),
    F.round(F.avg("is_first_txn"),        3).alias("first_txn_rate")
).orderBy("is_fraud").show()

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Part 4 — Ranking Functions
# MAGIC ### 4.1 row_number(), rank(), dense_rank()
# MAGIC
# MAGIC These are window functions that assign a number to each row within a partition.
# MAGIC They are slightly different from each other — important to know which to use.
# MAGIC
# MAGIC | Function | Ties | Gaps after ties |
# MAGIC |---|---|---|
# MAGIC | `row_number()` | Always unique — ties broken arbitrarily | No gaps |
# MAGIC | `rank()` | Tied rows get the same rank | Gaps after ties (1,1,3) |
# MAGIC | `dense_rank()` | Tied rows get the same rank | No gaps (1,1,2) |
# MAGIC
# MAGIC **Fraud use case:** find the largest transaction each user has ever made.

# COMMAND ----------

# Rank each user's transactions by amount (largest = rank 1)
rank_window = Window \
    .partitionBy("user_id") \
    .orderBy(F.col("amount").desc())

df_ranked = df.select(
    "transaction_id", "user_id", "amount", "timestamp", "is_fraud"
).withColumn("amount_rank",       F.rank().over(rank_window)) \
 .withColumn("amount_dense_rank", F.dense_rank().over(rank_window)) \
 .withColumn("amount_row_number", F.row_number().over(rank_window))

# Show top 3 transactions per user for a handful of users
df_ranked \
    .filter(F.col("amount_rank") <= 3) \
    .filter(F.col("user_id").isin("usr_0001", "usr_0002", "usr_0003")) \
    .orderBy("user_id", "amount_rank") \
    .show(20, truncate=False)

# COMMAND ----------

# Practical use: get each user's single largest transaction
df_largest_txn = df_ranked \
    .filter(F.col("amount_row_number") == 1) \
    .select("user_id", "transaction_id", "amount", "is_fraud") \
    .orderBy(F.col("amount").desc())

print(f"One row per user: {df_largest_txn.count():,} rows (should equal number of unique users)")
df_largest_txn.show(10, truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Part 5 — Composite Risk Score
# MAGIC ### 5.1 Combining Signals into a Single Score
# MAGIC
# MAGIC In production fraud systems, individual features are combined into a
# MAGIC **risk score** — a single number that summarises how suspicious a transaction is.
# MAGIC
# MAGIC We'll build a simple rule-based score by assigning points for each risk signal.
# MAGIC This is not a machine learning model — it's a **heuristic scorer**,
# MAGIC which is how many production systems start before ML is introduced.
# MAGIC
# MAGIC | Signal | Points |
# MAGIC |---|---|
# MAGIC | High-risk country (NG, RU, BR, CN) | +2 |
# MAGIC | Crypto payment method | +2 |
# MAGIC | Night-time transaction (midnight-5am) | +1 |
# MAGIC | Amount > 3x user's historical average | +2 |
# MAGIC | First-ever transaction for this user | +1 |
# MAGIC | High velocity: 3+ transactions in last hour | +2 |
# MAGIC | Very high amount (> 2000) | +1 |
# MAGIC
# MAGIC Max possible score: 11

# COMMAND ----------

HIGH_RISK_COUNTRIES = ["NG", "RU", "BR", "CN"]

df_scored = df_behavioural \
    .withColumn("hour_of_day", F.hour("timestamp")) \
    .withColumn(
        "risk_score",
        # High-risk country
        F.when(F.col("country").isin(HIGH_RISK_COUNTRIES), 2).otherwise(0)
        # Crypto payment
        + F.when(F.col("payment_method") == "crypto", 2).otherwise(0)
        # Night-time
        + F.when(F.col("hour_of_day") <= 5, 1).otherwise(0)
        # Amount > 3x user average
        + F.when(F.col("amount_vs_avg_ratio") > 3, 2).otherwise(0)
        # First transaction
        + F.when(F.col("is_first_txn") == 1, 1).otherwise(0)
        # High velocity
        + F.when(F.col("txn_count_1h") >= 3, 2).otherwise(0)
        # Very high amount
        + F.when(F.col("amount") > 2000, 1).otherwise(0)
    ) \
    .withColumn(
        "risk_label",
        F.when(F.col("risk_score") >= 5, "HIGH")
         .when(F.col("risk_score") >= 3, "MEDIUM")
         .otherwise("LOW")
    )

# Distribution of risk scores
print("=== Risk Score Distribution ===")
df_scored.groupBy("risk_score").agg(
    F.count("*").alias("transactions"),
    F.sum("is_fraud").alias("actual_fraud")
).orderBy("risk_score").show()

# COMMAND ----------

# How well does our risk label predict fraud?
print("=== Risk Label vs Actual Fraud ===")
df_scored.groupBy("risk_label").agg(
    F.count("*").alias("total_transactions"),
    F.sum("is_fraud").alias("actual_fraud_count"),
    F.round(F.sum("is_fraud") / F.count("*") * 100, 2).alias("fraud_rate_pct")
).orderBy("fraud_rate_pct").show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5.2 What to Look For
# MAGIC
# MAGIC A useful risk score should show:
# MAGIC - HIGH label → much higher fraud rate than average (~5%)
# MAGIC - LOW label → much lower fraud rate than average
# MAGIC
# MAGIC If HIGH has 20%+ fraud rate and LOW has <2%, our features have real signal.
# MAGIC This is called **feature discrimination** — how well a feature separates
# MAGIC fraud from legitimate transactions.

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Part 6 — Save the Feature Table
# MAGIC ### 6.1 Write to Delta
# MAGIC
# MAGIC We save the enriched feature table as a new Delta table.
# MAGIC This follows the **medallion architecture** pattern used in production:
# MAGIC
# MAGIC ```
# MAGIC Bronze  →  Raw ingested data          (fraud_transactions)
# MAGIC Silver  →  Cleaned + enriched data    (fraud_features)   ← we are here
# MAGIC Gold    →  Aggregated / ML-ready data (fraud_risk_scores) ← Notebook 4
# MAGIC ```

# COMMAND ----------

# Select the final feature set to save
# Drop intermediate columns we don't need downstream
feature_columns = [
    "transaction_id", "user_id", "merchant_id", "merchant_category",
    "amount", "country", "payment_method", "timestamp", "is_fraud",
    # Velocity features
    "txn_count_1h", "txn_count_24h", "txn_count_7d",
    "amount_sum_24h", "amount_sum_7d",
    # Behavioural features
    "user_avg_amount_historical", "amount_vs_avg_ratio",
    "user_txn_number", "is_first_txn",
    # Time features
    "hour_of_day",
    # Risk score
    "risk_score", "risk_label"
]

df_features = df_scored.select(feature_columns)

print(f"Feature table shape: {df_features.count():,} rows x {len(df_features.columns)} columns")
df_features.printSchema()

# COMMAND ----------

# Drop and recreate for clean reruns
spark.sql("DROP TABLE IF EXISTS workspace.default.fraud_features")

df_features.write \
    .format("delta") \
    .mode("overwrite") \
    .saveAsTable("workspace.default.fraud_features")

print("✅ Feature table written: workspace.default.fraud_features")

# Verify both tables exist
spark.sql("SHOW TABLES IN workspace.default").show()

# COMMAND ----------

# Final sanity check on the saved table
df_check = spark.table("workspace.default.fraud_features")
print(f"Rows: {df_check.count():,}")
df_check.show(5, truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## ✅ Notebook 3 Complete
# MAGIC
# MAGIC ### What you covered:
# MAGIC
# MAGIC | Topic | Key Takeaway |
# MAGIC |---|---|
# MAGIC | Window vs groupBy | Window keeps all rows; groupBy collapses them |
# MAGIC | `partitionBy()` | Defines the group — "do this per user" |
# MAGIC | `orderBy()` in window | Defines chronological order within each group |
# MAGIC | `rangeBetween()` | Time-based windows — last 1h, 24h, 7d |
# MAGIC | `rowsBetween()` | Row-count-based windows — all history to current row |
# MAGIC | Velocity features | Transaction count and spend in recent time windows |
# MAGIC | Behavioural features | How unusual is this transaction vs user's own history? |
# MAGIC | `row_number()` | Unique rank per row — use to get top-N per group |
# MAGIC | `rank()` / `dense_rank()` | Handle ties differently — know the difference |
# MAGIC | Composite risk score | Combine signals into one interpretable number |
# MAGIC | Medallion architecture | Bronze → Silver → Gold — the industry-standard pattern |
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### What's next — Notebook 4:
# MAGIC
# MAGIC - **Aggregations and Gold Layer** — build summary tables by user, merchant, country
# MAGIC - Joins in PySpark — enriching transactions with user/merchant profiles
# MAGIC - Writing production-quality PySpark with reusable functions
# MAGIC - Your first end-to-end pipeline: Bronze → Silver → Gold in one run