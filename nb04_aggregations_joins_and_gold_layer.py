# Databricks notebook source
# MAGIC %md
# MAGIC # 🏦 Payment Fraud Pipeline — Notebook 4
# MAGIC ## Aggregations, Joins & the Gold Layer
# MAGIC
# MAGIC **Project:** Payment Fraud Detection Pipeline
# MAGIC **Series:** Notebook 4 of N
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 🎯 What you'll learn in this notebook
# MAGIC
# MAGIC 1. Build **Gold layer summary tables** — aggregated views by user, merchant, and country
# MAGIC 2. Joins in PySpark — enriching transactions with profile-level context
# MAGIC 3. Join types: inner, left, anti — when to use each and why it matters
# MAGIC 4. Writing reusable PySpark functions — production code style
# MAGIC 5. Run your first **end-to-end pipeline**: Bronze → Silver → Gold in one cell
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 🧱 Where we are in the Medallion Architecture
# MAGIC
# MAGIC ```
# MAGIC Bronze  →  Raw ingested data          fraud_transactions   (Notebook 2)
# MAGIC Silver  →  Cleaned + enriched data    fraud_features       (Notebook 3)
# MAGIC Gold    →  Aggregated / joined data   fraud_user_profile   (this notebook)
# MAGIC                                       fraud_merchant_stats
# MAGIC                                       fraud_gold_enriched
# MAGIC ```
# MAGIC
# MAGIC Gold layer tables are what analysts, dashboards, and ML models consume.
# MAGIC They are pre-aggregated, joined, and ready to use — no heavy computation needed.

# COMMAND ----------

spark.sql("USE CATALOG workspace")
spark.sql("USE SCHEMA default")

spark.sql("SELECT current_catalog(), current_schema()").show()

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.window import Window

# Load the Silver layer feature table we built in Notebook 3
df = spark.table("workspace.default.fraud_features")

print(f"Loaded {df.count():,} rows from fraud_features")
df.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Part 1 — Gold Layer: User Profile Table
# MAGIC ### 1.1 What is a User Profile?
# MAGIC
# MAGIC A user profile aggregates everything we know about a user
# MAGIC across all their transactions. This is a classic Gold layer table.
# MAGIC
# MAGIC It answers questions like:
# MAGIC - How many transactions has this user made in total?
# MAGIC - What is their average spend?
# MAGIC - Have they had fraud before?
# MAGIC - Which countries and payment methods do they typically use?
# MAGIC
# MAGIC In production, this table is refreshed daily and joined onto every
# MAGIC new transaction to enrich it with user context.

# COMMAND ----------

# Build the user profile Gold table
df_user_profile = df.groupBy("user_id").agg(
    # Transaction counts
    F.count("*").alias("total_transactions"),
    F.sum("is_fraud").alias("total_fraud_txns"),
    F.round(F.sum("is_fraud") / F.count("*") * 100, 2).alias("historical_fraud_rate_pct"),

    # Spend profile
    F.round(F.avg("amount"), 2).alias("avg_transaction_amount"),
    F.round(F.min("amount"), 2).alias("min_transaction_amount"),
    F.round(F.max("amount"), 2).alias("max_transaction_amount"),
    F.round(F.sum("amount"), 2).alias("total_spend"),

    # Time profile
    F.min("timestamp").alias("first_transaction_date"),
    F.max("timestamp").alias("last_transaction_date"),

    # Behavioural flags
    F.round(F.avg("risk_score"), 2).alias("avg_risk_score"),
    F.max("risk_score").alias("max_risk_score"),
)

print(f"User profiles built: {df_user_profile.count():,} unique users")
df_user_profile.show(5, truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.2 Add a User Risk Tier
# MAGIC
# MAGIC Based on their historical fraud rate and average risk score,
# MAGIC we classify each user into a risk tier.
# MAGIC This is a common pattern in financial services — customer risk segmentation.

# COMMAND ----------

df_user_profile = df_user_profile.withColumn(
    "user_risk_tier",
    F.when(
        (F.col("historical_fraud_rate_pct") > 10) | (F.col("avg_risk_score") >= 4),
        "HIGH"
    ).when(
        (F.col("historical_fraud_rate_pct") > 3) | (F.col("avg_risk_score") >= 2),
        "MEDIUM"
    ).otherwise("LOW")
)

# Distribution of user risk tiers
df_user_profile.groupBy("user_risk_tier").agg(
    F.count("*").alias("user_count"),
    F.round(F.avg("total_transactions"), 1).alias("avg_txns"),
    F.round(F.avg("avg_transaction_amount"), 2).alias("avg_spend"),
    F.round(F.avg("historical_fraud_rate_pct"), 2).alias("avg_fraud_rate_pct")
).orderBy("avg_fraud_rate_pct").show()

# COMMAND ----------

# Save user profile as Gold table
spark.sql("DROP TABLE IF EXISTS workspace.default.fraud_user_profile")

df_user_profile.write \
    .format("delta") \
    .mode("overwrite") \
    .saveAsTable("workspace.default.fraud_user_profile")

print("✅ Saved: workspace.default.fraud_user_profile")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Part 2 — Gold Layer: Merchant Stats Table
# MAGIC ### 2.1 Merchant Risk Profiles
# MAGIC
# MAGIC Merchants are the other side of every transaction.
# MAGIC Some merchants attract more fraud than others — either because they sell
# MAGIC high-value goods (electronics, luxury) or because their verification is weak.
# MAGIC
# MAGIC This table gives us a risk profile for every merchant.

# COMMAND ----------

df_merchant_stats = df.groupBy("merchant_id", "merchant_category").agg(
    F.count("*").alias("total_transactions"),
    F.sum("is_fraud").alias("total_fraud_txns"),
    F.round(F.sum("is_fraud") / F.count("*") * 100, 2).alias("merchant_fraud_rate_pct"),
    F.round(F.avg("amount"), 2).alias("avg_transaction_amount"),
    F.round(F.sum("amount"), 2).alias("total_volume"),
    F.countDistinct("user_id").alias("unique_customers"),
    F.round(F.avg("risk_score"), 2).alias("avg_risk_score")
).withColumn(
    "merchant_risk_tier",
    F.when(F.col("merchant_fraud_rate_pct") > 10, "HIGH")
     .when(F.col("merchant_fraud_rate_pct") > 3,  "MEDIUM")
     .otherwise("LOW")
)

print(f"Merchant profiles built: {df_merchant_stats.count():,} merchants")

# Fraud rate by merchant category — which categories attract most fraud?
print("\n=== Fraud rate by merchant category ===")
df_merchant_stats.groupBy("merchant_category").agg(
    F.round(F.avg("merchant_fraud_rate_pct"), 2).alias("avg_fraud_rate_pct"),
    F.count("*").alias("merchant_count")
).orderBy(F.col("avg_fraud_rate_pct").desc()).show()

# COMMAND ----------

# Save merchant stats as Gold table
spark.sql("DROP TABLE IF EXISTS workspace.default.fraud_merchant_stats")

df_merchant_stats.write \
    .format("delta") \
    .mode("overwrite") \
    .saveAsTable("workspace.default.fraud_merchant_stats")

print("✅ Saved: workspace.default.fraud_merchant_stats")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Part 3 — Joins in PySpark
# MAGIC ### 3.1 Join Types — Know Which to Use
# MAGIC
# MAGIC PySpark supports all standard SQL join types. The three you'll use most:
# MAGIC
# MAGIC | Join Type | Returns | Use when |
# MAGIC |---|---|---|
# MAGIC | `inner` | Only rows with a match in BOTH tables | You need data from both sides |
# MAGIC | `left` | All rows from left, matched from right (NULL if no match) | You want to keep all left rows |
# MAGIC | `anti` | Rows in left that have NO match in right | Finding gaps / unmatched records |
# MAGIC
# MAGIC **The golden rule:** always use `left` join unless you have a specific reason
# MAGIC for `inner`. An `inner` join silently drops rows with no match — in a fraud
# MAGIC pipeline, silently dropping transactions is dangerous.

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.2 Enrich Transactions with User Profile
# MAGIC
# MAGIC We join each transaction (Silver) with the user's profile (Gold)
# MAGIC to create a fully enriched transaction record.
# MAGIC
# MAGIC This is the most common join pattern in data engineering:
# MAGIC **fact table (transactions) LEFT JOIN dimension table (user profile)**

# COMMAND ----------

# Left join: keep ALL transactions, bring in user profile columns
df_enriched = df.alias("txn").join(
    df_user_profile.select(
        "user_id",
        "user_risk_tier",
        "avg_transaction_amount",
        "total_transactions",
        "historical_fraud_rate_pct"
    ).alias("usr"),
    on="user_id",
    how="left"
)

print(f"Transactions before join: {df.count():,}")
print(f"Transactions after join:  {df_enriched.count():,}")
print("Row count should be identical — left join keeps all left rows")

# COMMAND ----------

# Also join merchant stats
df_enriched = df_enriched.join(
    df_merchant_stats.select(
        "merchant_id",
        "merchant_risk_tier",
        "merchant_fraud_rate_pct"
    ).alias("mrc"),
    on="merchant_id",
    how="left"
)

# Preview the enriched result
df_enriched.select(
    "transaction_id", "user_id", "amount", "is_fraud",
    "user_risk_tier", "merchant_risk_tier",
    "historical_fraud_rate_pct", "merchant_fraud_rate_pct",
    "risk_score", "risk_label"
).show(10, truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.3 Anti Join — Finding Transactions with No User Profile
# MAGIC
# MAGIC An anti join returns rows from the left table that have NO match
# MAGIC in the right table. In fraud detection, this is useful for finding
# MAGIC "ghost" transactions — activity with no known user profile.

# COMMAND ----------

# Find any transactions where the user has no profile (should be zero — just demonstrating the pattern)
df_no_profile = df.join(
    df_user_profile.select("user_id"),
    on="user_id",
    how="anti"
)

print(f"Transactions with no user profile: {df_no_profile.count():,}")
print("(Zero expected — every user in transactions also has a profile)")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Part 4 — Reusable Functions
# MAGIC ### 4.1 Writing Production-Style PySpark
# MAGIC
# MAGIC So far we've written all our transformations inline.
# MAGIC In production pipelines, you wrap logic into functions so it is:
# MAGIC - **Reusable** — call the same function in multiple notebooks
# MAGIC - **Testable** — unit test each transformation independently
# MAGIC - **Readable** — a pipeline reads like a story, not a wall of code
# MAGIC
# MAGIC The pattern: each function takes a DataFrame and returns a DataFrame.
# MAGIC This is called a **transformation function** — very standard in PySpark.

# COMMAND ----------

def add_combined_risk_tier(df):
    """
    Combines user risk tier and merchant risk tier into a single
    combined risk assessment for the transaction.

    A transaction is HIGH risk if EITHER the user OR merchant is HIGH risk.
    """
    return df.withColumn(
        "combined_risk_tier",
        F.when(
            (F.col("user_risk_tier") == "HIGH") | (F.col("merchant_risk_tier") == "HIGH"),
            "HIGH"
        ).when(
            (F.col("user_risk_tier") == "MEDIUM") | (F.col("merchant_risk_tier") == "MEDIUM"),
            "MEDIUM"
        ).otherwise("LOW")
    )


def add_time_features(df):
    """
    Adds hour, day of week, and weekend flag from timestamp.
    Centralising time features in one function makes them easy to update.
    """
    return df \
        .withColumn("hour_of_day",  F.hour("timestamp")) \
        .withColumn("day_of_week",  F.dayofweek("timestamp")) \
        .withColumn("is_weekend",   (F.dayofweek("timestamp").isin(1, 7)).cast("int"))


def add_amount_band(df):
    """
    Buckets transaction amounts into bands.
    Useful for reporting and for categorical ML features.
    """
    return df.withColumn(
        "amount_band",
        F.when(F.col("amount") < 50,   "micro")
         .when(F.col("amount") < 200,  "small")
         .when(F.col("amount") < 1000, "medium")
         .when(F.col("amount") < 5000, "large")
         .otherwise("very_large")
    )


print("Transformation functions defined.")

# COMMAND ----------

# Apply all transformations in a clean pipeline chain
df_gold = df_enriched \
    .transform(add_combined_risk_tier) \
    .transform(add_time_features) \
    .transform(add_amount_band)

# Verify
df_gold.select(
    "transaction_id", "amount", "amount_band",
    "user_risk_tier", "merchant_risk_tier", "combined_risk_tier",
    "is_weekend", "is_fraud"
).show(10, truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4.2 Does the Combined Risk Tier Predict Fraud?

# COMMAND ----------

df_gold.groupBy("combined_risk_tier").agg(
    F.count("*").alias("total_transactions"),
    F.sum("is_fraud").alias("actual_fraud"),
    F.round(F.sum("is_fraud") / F.count("*") * 100, 2).alias("fraud_rate_pct")
).orderBy("fraud_rate_pct").show()

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Part 5 — End-to-End Pipeline Function
# MAGIC ### 5.1 Bronze → Silver → Gold in One Function
# MAGIC
# MAGIC This is the payoff of writing modular, reusable functions.
# MAGIC We can now express the entire pipeline as a clean sequence of steps.
# MAGIC
# MAGIC In production, this function would be scheduled to run daily —
# MAGIC reading yesterday's new transactions and producing fresh Gold tables.

# COMMAND ----------

def run_gold_pipeline(spark):
    """
    End-to-end pipeline: reads Silver feature table,
    joins user and merchant profiles, applies enrichments,
    and writes the Gold enriched table.

    Returns the final Gold DataFrame.
    """
    print("📥 Step 1: Load Silver feature table...")
    df_silver = spark.table("workspace.default.fraud_features")

    print("📥 Step 2: Load dimension tables...")
    df_users     = spark.table("workspace.default.fraud_user_profile")
    df_merchants = spark.table("workspace.default.fraud_merchant_stats")

    print("🔗 Step 3: Join user profiles...")
    df = df_silver.join(
        df_users.select("user_id", "user_risk_tier", "historical_fraud_rate_pct"),
        on="user_id", how="left"
    )

    print("🔗 Step 4: Join merchant profiles...")
    df = df.join(
        df_merchants.select("merchant_id", "merchant_risk_tier", "merchant_fraud_rate_pct"),
        on="merchant_id", how="left"
    )

    print("🔧 Step 5: Apply enrichment transformations...")
    df = df \
        .transform(add_combined_risk_tier) \
        .transform(add_time_features) \
        .transform(add_amount_band)

    print("💾 Step 6: Write Gold table...")
    spark.sql("DROP TABLE IF EXISTS workspace.default.fraud_gold_enriched")
    df.write \
        .format("delta") \
        .mode("overwrite") \
        .saveAsTable("workspace.default.fraud_gold_enriched")

    row_count = df.count()
    print(f"\n✅ Pipeline complete. Gold table: {row_count:,} rows")
    return df


# Run it
df_gold_final = run_gold_pipeline(spark)

# COMMAND ----------

# Final check — all three Gold tables should exist now
spark.sql("SHOW TABLES IN workspace.default").show()

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## ✅ Notebook 4 Complete
# MAGIC
# MAGIC ### What you covered:
# MAGIC
# MAGIC | Topic | Key Takeaway |
# MAGIC |---|---|
# MAGIC | Gold layer tables | Pre-aggregated, join-ready — what analysts and ML models consume |
# MAGIC | User profile table | One row per user summarising all their transaction history |
# MAGIC | Merchant stats table | Fraud rates and risk tiers per merchant and category |
# MAGIC | `left` join | Always the default — never silently drop rows |
# MAGIC | `inner` join | Only use when you explicitly want to drop non-matching rows |
# MAGIC | `anti` join | Find gaps — rows in left with no match in right |
# MAGIC | Transformation functions | `def f(df): return df.withColumn(...)` — reusable, testable |
# MAGIC | `.transform()` | Chain functions cleanly — reads like a pipeline story |
# MAGIC | End-to-end pipeline | Bronze → Silver → Gold expressed as a single function |
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### What's next — Notebook 5:
# MAGIC
# MAGIC - **Monitoring & Data Quality** — production pipelines need guardrails
# MAGIC - Row count checks, null checks, fraud rate drift detection
# MAGIC - Writing a reusable data quality checker in PySpark
# MAGIC - Introduction to **Delta constraints** — enforce rules at the table level
# MAGIC - Wrapping everything up: your complete portfolio pipeline