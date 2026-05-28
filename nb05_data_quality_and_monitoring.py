# Databricks notebook source
# MAGIC %md
# MAGIC # 🏦 Payment Fraud Pipeline — Notebook 5
# MAGIC ## Data Quality & Monitoring
# MAGIC
# MAGIC **Project:** Payment Fraud Detection Pipeline
# MAGIC **Series:** Notebook 5 of N
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 🎯 What you'll learn in this notebook
# MAGIC
# MAGIC 1. Why data quality is non-negotiable in production pipelines
# MAGIC 2. Build a reusable **data quality checker** class in PySpark
# MAGIC 3. Row count checks, null checks, range checks, uniqueness checks
# MAGIC 4. Fraud rate drift detection — catch when your data changes unexpectedly
# MAGIC 5. Delta table **constraints** — enforce rules at the storage level
# MAGIC 6. A monitoring summary report across all pipeline tables
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 🧠 Why data quality matters in fraud pipelines
# MAGIC
# MAGIC Imagine your fraud pipeline silently receives a feed where:
# MAGIC - 30% of `amount` values are suddenly NULL
# MAGIC - The fraud rate drops from 5% to 0.1% overnight
# MAGIC - Duplicate transaction IDs start appearing
# MAGIC
# MAGIC Without quality checks, your pipeline runs, produces results, and nobody
# MAGIC notices until a fraud analyst asks "why did we miss 500 fraud cases this week?"
# MAGIC
# MAGIC Data quality checks are the **safety net** between raw data and business decisions.
# MAGIC In interviews, mentioning that your pipeline includes DQ checks immediately
# MAGIC signals production-level thinking.

# COMMAND ----------

spark.sql("USE CATALOG workspace")
spark.sql("USE SCHEMA default")

from pyspark.sql import functions as F
from pyspark.sql import DataFrame
from typing import List, Dict, Any
from datetime import datetime

spark.sql("SELECT current_catalog(), current_schema()").show()

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Part 1 — Building a Data Quality Checker
# MAGIC ### 1.1 The Design
# MAGIC
# MAGIC We'll build a `DataQualityChecker` class that:
# MAGIC - Takes a DataFrame and a table name
# MAGIC - Runs a suite of checks (row count, nulls, ranges, uniqueness)
# MAGIC - Returns a results summary — pass ✅ or fail ❌ for each check
# MAGIC - Raises an error if any critical check fails
# MAGIC
# MAGIC This is a real pattern used in production pipelines at companies like
# MAGIC Revolut, Monzo, and Stripe — sometimes called a "data contract" checker.

# COMMAND ----------

class DataQualityChecker:
    """
    Reusable data quality checker for PySpark DataFrames.
    Each check returns a dict with: check_name, status, details.
    """

    def __init__(self, df: DataFrame, table_name: str):
        self.df = df
        self.table_name = table_name
        self.results = []
        self._row_count = None  # cache so we don't call .count() multiple times

    @property
    def row_count(self):
        if self._row_count is None:
            self._row_count = self.df.count()
        return self._row_count

    def _record(self, check_name: str, passed: bool, details: str):
        status = "✅ PASS" if passed else "❌ FAIL"
        self.results.append({
            "table":      self.table_name,
            "check":      check_name,
            "status":     status,
            "details":    details
        })
        print(f"  {status}  {check_name}: {details}")
        return self

    # ------------------------------------------------------------------
    # Checks
    # ------------------------------------------------------------------

    def check_row_count(self, min_rows: int, max_rows: int = None):
        """Verify row count is within expected bounds."""
        count = self.row_count
        if max_rows:
            passed = min_rows <= count <= max_rows
            details = f"{count:,} rows (expected {min_rows:,} – {max_rows:,})"
        else:
            passed = count >= min_rows
            details = f"{count:,} rows (expected >= {min_rows:,})"
        return self._record("row_count", passed, details)

    def check_no_nulls(self, columns: List[str]):
        """Verify specified columns have zero null values."""
        for col in columns:
            null_count = self.df.filter(F.col(col).isNull()).count()
            passed = null_count == 0
            details = f"column '{col}': {null_count:,} nulls"
            self._record(f"no_nulls.{col}", passed, details)
        return self

    def check_uniqueness(self, columns: List[str]):
        """Verify specified columns have no duplicate values."""
        for col in columns:
            total     = self.row_count
            distinct  = self.df.select(col).distinct().count()
            dupes     = total - distinct
            passed    = dupes == 0
            details   = f"column '{col}': {dupes:,} duplicates ({distinct:,} distinct / {total:,} total)"
            self._record(f"uniqueness.{col}", passed, details)
        return self

    def check_value_range(self, column: str, min_val: float, max_val: float):
        """Verify all values in a column fall within [min_val, max_val]."""
        out_of_range = self.df.filter(
            (F.col(column) < min_val) | (F.col(column) > max_val)
        ).count()
        passed  = out_of_range == 0
        details = f"column '{column}': {out_of_range:,} values outside [{min_val}, {max_val}]"
        return self._record(f"value_range.{column}", passed, details)

    def check_allowed_values(self, column: str, allowed: List[str]):
        """Verify a column only contains values from an allowed set."""
        invalid = self.df.filter(~F.col(column).isin(allowed)).count()
        passed  = invalid == 0
        details = f"column '{column}': {invalid:,} values not in allowed set {allowed}"
        return self._record(f"allowed_values.{column}", passed, details)

    def check_metric_in_range(self, metric_name: str, actual: float,
                               min_val: float, max_val: float):
        """Verify a computed metric (e.g. fraud rate) is within expected bounds."""
        passed  = min_val <= actual <= max_val
        details = f"{metric_name}: {actual:.4f} (expected {min_val} – {max_val})"
        return self._record(f"metric.{metric_name}", passed, details)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> DataFrame:
        """Return results as a Spark DataFrame for easy display."""
        return spark.createDataFrame(self.results)

    def assert_no_failures(self):
        """Raise an exception if any check failed — use to halt a pipeline."""
        failures = [r for r in self.results if "FAIL" in r["status"]]
        if failures:
            msg = "\n".join([f"  {f['check']}: {f['details']}" for f in failures])
            raise AssertionError(f"\n❌ Data quality failures in {self.table_name}:\n{msg}")
        print(f"\n✅ All checks passed for {self.table_name}")


print("DataQualityChecker class defined.")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Part 2 — Run Checks on Each Pipeline Table
# MAGIC ### 2.1 Bronze Table: fraud_transactions

# COMMAND ----------

print("=" * 60)
print("CHECKING: workspace.default.fraud_transactions")
print("=" * 60)

df_bronze = spark.table("workspace.default.fraud_transactions")

bronze_checker = DataQualityChecker(df_bronze, "fraud_transactions")
bronze_checker \
    .check_row_count(min_rows=9_000, max_rows=11_000) \
    .check_no_nulls(["transaction_id", "user_id", "amount", "timestamp", "is_fraud"]) \
    .check_uniqueness(["transaction_id"]) \
    .check_value_range("amount", min_val=0.01, max_val=20_000) \
    .check_value_range("is_fraud", min_val=0, max_val=1) \
    .check_allowed_values("payment_method", ["card", "bank_transfer", "wallet", "crypto", "voucher"]) \
    .check_allowed_values("country", ["GB", "US", "JP", "DE", "FR", "NG", "RU", "BR", "CN", "IN"])

# Check fraud rate is within expected range (3% – 8%)
fraud_rate = df_bronze.agg(F.avg("is_fraud")).collect()[0][0]
bronze_checker.check_metric_in_range("fraud_rate", fraud_rate, 0.03, 0.08)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.2 Silver Table: fraud_features

# COMMAND ----------

print("=" * 60)
print("CHECKING: workspace.default.fraud_features")
print("=" * 60)

df_silver = spark.table("workspace.default.fraud_features")

silver_checker = DataQualityChecker(df_silver, "fraud_features")
silver_checker \
    .check_row_count(min_rows=9_000, max_rows=11_000) \
    .check_no_nulls(["transaction_id", "user_id", "risk_score", "risk_label"]) \
    .check_uniqueness(["transaction_id"]) \
    .check_value_range("risk_score", min_val=0, max_val=11) \
    .check_value_range("txn_count_1h", min_val=1, max_val=500) \
    .check_allowed_values("risk_label", ["LOW", "MEDIUM", "HIGH"])

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.3 Gold Table: fraud_user_profile

# COMMAND ----------

print("=" * 60)
print("CHECKING: workspace.default.fraud_user_profile")
print("=" * 60)

df_users = spark.table("workspace.default.fraud_user_profile")

user_checker = DataQualityChecker(df_users, "fraud_user_profile")
user_checker \
    .check_row_count(min_rows=400, max_rows=600) \
    .check_no_nulls(["user_id", "total_transactions", "avg_transaction_amount"]) \
    .check_uniqueness(["user_id"]) \
    .check_value_range("historical_fraud_rate_pct", min_val=0, max_val=100) \
    .check_allowed_values("user_risk_tier", ["LOW", "MEDIUM", "HIGH"])

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.4 Gold Table: fraud_merchant_stats

# COMMAND ----------

print("=" * 60)
print("CHECKING: workspace.default.fraud_merchant_stats")
print("=" * 60)

df_merchants = spark.table("workspace.default.fraud_merchant_stats")

merchant_checker = DataQualityChecker(df_merchants, "fraud_merchant_stats")
merchant_checker \
    .check_row_count(min_rows=1_500, max_rows=2_500) \
    .check_no_nulls(["merchant_id", "merchant_category", "total_transactions"]) \
    .check_uniqueness(["merchant_id"]) \
    .check_value_range("merchant_fraud_rate_pct", min_val=0, max_val=100) \
    .check_allowed_values("merchant_risk_tier", ["LOW", "MEDIUM", "HIGH"])

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Part 3 — Fraud Rate Drift Detection
# MAGIC ### 3.1 What is Drift?
# MAGIC
# MAGIC **Drift** means your data is changing in unexpected ways over time.
# MAGIC In fraud detection, drift is critical to monitor:
# MAGIC
# MAGIC - **Fraud rate drops suddenly** → maybe fraud is being missed, or data feed is broken
# MAGIC - **Fraud rate spikes suddenly** → possible attack wave, or a data pipeline bug
# MAGIC - **Average amount changes** → currency conversion issue, or data source changed
# MAGIC
# MAGIC We detect drift by comparing metrics across time windows —
# MAGIC specifically, comparing recent data against historical baseline.

# COMMAND ----------

# Add a month column for time-based analysis
df_bronze_monthly = df_bronze.withColumn("month", F.month("timestamp"))

# Calculate fraud rate and average amount per month
df_monthly_stats = df_bronze_monthly.groupBy("month").agg(
    F.count("*").alias("total_transactions"),
    F.sum("is_fraud").alias("fraud_count"),
    F.round(F.sum("is_fraud") / F.count("*") * 100, 3).alias("fraud_rate_pct"),
    F.round(F.avg("amount"), 2).alias("avg_amount")
).orderBy("month")

print("=== Monthly fraud rate (drift check) ===")
df_monthly_stats.show(12, truncate=False)

# COMMAND ----------

# Flag months where fraud rate deviates significantly from the annual average
overall_fraud_rate = df_bronze.agg(F.avg("is_fraud")).collect()[0][0] * 100

print(f"Overall fraud rate: {overall_fraud_rate:.2f}%")
print(f"Expected range:     {overall_fraud_rate * 0.5:.2f}% – {overall_fraud_rate * 1.5:.2f}%\n")

df_monthly_stats.withColumn(
    "drift_flag",
    F.when(
        (F.col("fraud_rate_pct") < overall_fraud_rate * 0.5) |
        (F.col("fraud_rate_pct") > overall_fraud_rate * 1.5),
        "⚠️  DRIFT DETECTED"
    ).otherwise("✅ normal")
).show(12, truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Part 4 — Delta Table Constraints
# MAGIC ### 4.1 Enforcing Rules at the Storage Level
# MAGIC
# MAGIC So far our quality checks run at pipeline time — they catch bad data
# MAGIC after it arrives. Delta constraints are different: they **prevent**
# MAGIC bad data from being written to the table in the first place.
# MAGIC
# MAGIC This is the difference between:
# MAGIC - **Defensive checking** (our DataQualityChecker) — catch bad data, alert, decide what to do
# MAGIC - **Enforcement** (Delta constraints) — reject bad data at write time, always
# MAGIC
# MAGIC Both are useful. Use constraints for rules that must NEVER be violated.

# COMMAND ----------

# Add constraints to the fraud_transactions table
# These will reject any future write that violates them

# Constraint: amount must always be positive
spark.sql("""
    ALTER TABLE workspace.default.fraud_transactions
    ADD CONSTRAINT amount_positive CHECK (amount > 0)
""")

# Constraint: is_fraud must be 0 or 1
spark.sql("""
    ALTER TABLE workspace.default.fraud_transactions
    ADD CONSTRAINT is_fraud_binary CHECK (is_fraud = 0 OR is_fraud = 1)
""")

print("✅ Constraints added to fraud_transactions")

# COMMAND ----------

# View the constraints on the table
spark.sql("DESCRIBE DETAIL workspace.default.fraud_transactions") \
    .select("name", "properties") \
    .show(truncate=False)


# COMMAND ----------

# MAGIC %md
# MAGIC ### 4.2 Test That Constraints Work
# MAGIC
# MAGIC Let's try to insert a row with a negative amount — Delta should reject it.

# COMMAND ----------

# Try to write a bad row — this SHOULD fail with a constraint violation
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType, TimestampType
from datetime import datetime

bad_row = spark.createDataFrame([
    ("txn_BAD", "usr_0001", "mrc_0001", "electronics", -50.0, "GB", "card",
     datetime(2024, 6, 1, 12, 0, 0), 0)
], ["transaction_id", "user_id", "merchant_id", "merchant_category",
    "amount", "country", "payment_method", "timestamp", "is_fraud"])

try:
    bad_row.write \
        .format("delta") \
        .mode("append") \
        .saveAsTable("workspace.default.fraud_transactions")
    print("❌ Write succeeded — constraint did NOT fire (unexpected)")
except Exception as e:
    print("✅ Constraint worked — bad row was rejected")
    print(f"   Error: {str(e)[:120]}...")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Part 5 — Pipeline Monitoring Summary
# MAGIC ### 5.1 Combine All Check Results into One Report

# COMMAND ----------

# Combine all checker results into one summary DataFrame
df_all_results = bronze_checker.summary() \
    .union(silver_checker.summary()) \
    .union(user_checker.summary()) \
    .union(merchant_checker.summary())

print("=" * 60)
print("FULL PIPELINE DATA QUALITY REPORT")
print("=" * 60)
df_all_results.show(100, truncate=False)

# COMMAND ----------

# Summary by table — how many passed vs failed per table
print("=== Summary by table ===")
df_all_results.groupBy("table").agg(
    F.sum(F.when(F.col("status").contains("PASS"), 1).otherwise(0)).alias("passed"),
    F.sum(F.when(F.col("status").contains("FAIL"), 1).otherwise(0)).alias("failed"),
    F.count("*").alias("total_checks")
).show(truncate=False)

# COMMAND ----------

# Overall pipeline health
total_checks  = df_all_results.count()
failed_checks = df_all_results.filter(F.col("status").contains("FAIL")).count()
passed_checks = total_checks - failed_checks

print(f"\n{'='*40}")
print(f"PIPELINE HEALTH SUMMARY")
print(f"{'='*40}")
print(f"Total checks:  {total_checks}")
print(f"Passed:        {passed_checks} ✅")
print(f"Failed:        {failed_checks} {'❌' if failed_checks > 0 else '✅'}")
print(f"Pass rate:     {passed_checks/total_checks*100:.1f}%")
print(f"{'='*40}")

# COMMAND ----------

# Save the DQ report as a Delta table — useful for trending quality over time
spark.sql("DROP TABLE IF EXISTS workspace.default.fraud_dq_report")

df_all_results \
    .withColumn("report_timestamp", F.current_timestamp()) \
    .write \
    .format("delta") \
    .mode("overwrite") \
    .saveAsTable("workspace.default.fraud_dq_report")

print("✅ DQ report saved: workspace.default.fraud_dq_report")

# COMMAND ----------

# Final: all pipeline tables
spark.sql("SHOW TABLES IN workspace.default").show()

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## ✅ Notebook 5 Complete
# MAGIC
# MAGIC ### What you covered:
# MAGIC
# MAGIC | Topic | Key Takeaway |
# MAGIC |---|---|
# MAGIC | Why data quality matters | Silent failures are worse than loud ones |
# MAGIC | `DataQualityChecker` class | Reusable, chainable — production-grade pattern |
# MAGIC | Row count checks | Catch dropped or duplicated data early |
# MAGIC | Null checks | Critical columns must never be null |
# MAGIC | Uniqueness checks | Transaction IDs must be unique — duplicates break aggregations |
# MAGIC | Range checks | Amounts can't be negative; fraud flag must be 0 or 1 |
# MAGIC | Allowed values checks | Payment methods must come from a known set |
# MAGIC | Metric checks | Fraud rate must stay within a plausible range |
# MAGIC | Drift detection | Compare monthly metrics against historical baseline |
# MAGIC | Delta constraints | Enforce rules at write time — reject bad data permanently |
# MAGIC | DQ report table | Persist check results — trend quality over time |
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 🏁 Your complete pipeline tables:
# MAGIC
# MAGIC | Table | Layer | Description |
# MAGIC |---|---|---|
# MAGIC | `fraud_transactions` | Bronze | Raw synthetic transactions |
# MAGIC | `fraud_features` | Silver | Enriched with window function features |
# MAGIC | `fraud_user_profile` | Gold | One row per user — risk tier and spend profile |
# MAGIC | `fraud_merchant_stats` | Gold | One row per merchant — fraud rates by category |
# MAGIC | `fraud_gold_enriched` | Gold | Fully joined and enriched transaction table |
# MAGIC | `fraud_dq_report` | Monitoring | Data quality check results |
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### What's next — Notebook 6 (GitHub & Portfolio):
# MAGIC
# MAGIC - Clean up and finalise all 5 notebooks for GitHub
# MAGIC - Write a professional `README.md` for the repository
# MAGIC - Structure the repo like a real data engineering portfolio project