# Databricks notebook source
# MAGIC %md
# MAGIC # 🏦 Payment Fraud Pipeline — Notebook 1
# MAGIC ## Python & Pandas Refresher + Spark Orientation
# MAGIC
# MAGIC **Project:** Payment Fraud Detection Pipeline
# MAGIC **Series:** Notebook 1 of N
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 🎯 What you'll learn in this notebook
# MAGIC
# MAGIC 1. Python fundamentals you'll use constantly in data work (lists, dicts, list comprehensions, lambdas)
# MAGIC 2. Pandas basics — creating DataFrames, filtering, grouping, aggregating
# MAGIC 3. What Spark is and why it exists (the "why" before the "how")
# MAGIC 4. Your first PySpark DataFrame — creating it, inspecting it, comparing it to pandas
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 📦 The Dataset We're Building Toward
# MAGIC
# MAGIC Throughout this project, we're going to work with a **simulated payment transactions dataset** — the kind of data a fintech company like Revolut, Stripe, or PayPal would have.
# MAGIC
# MAGIC Each row will represent a single payment transaction with fields like:
# MAGIC
# MAGIC | Field | Description |
# MAGIC |---|---|
# MAGIC | `transaction_id` | Unique ID for the transaction |
# MAGIC | `user_id` | The customer who made the payment |
# MAGIC | `merchant_id` | The merchant receiving the payment |
# MAGIC | `amount` | Transaction amount in USD |
# MAGIC | `timestamp` | When the transaction happened |
# MAGIC | `country` | Country of the transaction |
# MAGIC | `payment_method` | e.g. card, bank_transfer, wallet |
# MAGIC | `is_fraud` | 1 = fraudulent, 0 = legitimate |
# MAGIC
# MAGIC We'll generate this data ourselves in Notebook 2. For now, we'll work with small toy examples to build intuition.

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Part 1 — Python Refresher
# MAGIC ### 1.1 Lists and Dictionaries
# MAGIC
# MAGIC These two data structures are the backbone of almost every data pipeline you'll write.
# MAGIC
# MAGIC - A **list** is an ordered collection of items: `[1, 2, 3]`
# MAGIC - A **dict** is a key-value store: `{"name": "Nathan", "country": "JP"}`
# MAGIC
# MAGIC In data work, you'll use dicts constantly to represent a single row of data,
# MAGIC and lists to hold many rows before loading into a DataFrame.

# COMMAND ----------

# A single transaction represented as a dictionary
transaction = {
    "transaction_id": "txn_001",
    "user_id": "usr_42",
    "amount": 249.99,
    "country": "GB",
    "is_fraud": 0
}

# Access a value by key
print("Amount:", transaction["amount"])
print("Country:", transaction["country"])

# COMMAND ----------

# A list of transactions (list of dicts) — this is how you'll often seed DataFrames
transactions = [
    {"transaction_id": "txn_001", "user_id": "usr_42", "amount": 249.99, "country": "GB", "is_fraud": 0},
    {"transaction_id": "txn_002", "user_id": "usr_07", "amount": 8500.00, "country": "NG", "is_fraud": 1},
    {"transaction_id": "txn_003", "user_id": "usr_42", "amount": 15.50,   "country": "GB", "is_fraud": 0},
    {"transaction_id": "txn_004", "user_id": "usr_19", "amount": 3200.00, "country": "RU", "is_fraud": 1},
    {"transaction_id": "txn_005", "user_id": "usr_88", "amount": 55.00,   "country": "JP", "is_fraud": 0},
]

print(f"Number of transactions: {len(transactions)}")
print(f"First transaction: {transactions[0]}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.2 List Comprehensions
# MAGIC
# MAGIC A list comprehension is a concise way to build a new list by looping over an existing one.
# MAGIC
# MAGIC **Pattern:** `[expression for item in iterable if condition]`
# MAGIC
# MAGIC You'll use these constantly in PySpark work — for building column lists,
# MAGIC transforming field names, filtering things, etc.

# COMMAND ----------

# Extract just the amounts from all transactions
amounts = [t["amount"] for t in transactions]
print("All amounts:", amounts)

# Extract only the fraudulent transaction IDs
fraud_ids = [t["transaction_id"] for t in transactions if t["is_fraud"] == 1]
print("Fraud transaction IDs:", fraud_ids)

# Flag high-value transactions (over 1000)
high_value_flags = ["HIGH" if t["amount"] > 1000 else "normal" for t in transactions]
print("Value flags:", high_value_flags)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.3 Lambda Functions
# MAGIC
# MAGIC A **lambda** is a small, anonymous (unnamed) function written in one line.
# MAGIC
# MAGIC **Pattern:** `lambda argument: expression`
# MAGIC
# MAGIC In data work, lambdas are most often used with `map()`, `filter()`, `sorted()`,
# MAGIC and with pandas `.apply()`.

# COMMAND ----------

# A regular function vs a lambda — same result
def flag_amount(amount):
    return "HIGH" if amount > 1000 else "normal"

flag_amount_lambda = lambda amount: "HIGH" if amount > 1000 else "normal"

print(flag_amount(8500))         # using regular function
print(flag_amount_lambda(8500))  # using lambda — same result

# COMMAND ----------

# Practical use: sort transactions by amount (descending)
sorted_transactions = sorted(transactions, key=lambda t: t["amount"], reverse=True)

for t in sorted_transactions:
    print(f"{t['transaction_id']} — ${t['amount']}")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Part 2 — Pandas Refresher
# MAGIC ### 2.1 Creating a DataFrame
# MAGIC
# MAGIC Pandas is Python's primary in-memory data table library.
# MAGIC Think of a pandas DataFrame as an Excel spreadsheet in code.
# MAGIC
# MAGIC You can create one from:
# MAGIC - A list of dicts (most common in practice)
# MAGIC - A CSV file
# MAGIC - A database query result
# MAGIC
# MAGIC **Important context:** In this project, pandas is our "small data scratchpad".
# MAGIC When the data gets big, we switch to PySpark. You'll see exactly why later.

# COMMAND ----------

import pandas as pd

# Create a pandas DataFrame from our list of dicts
df_pandas = pd.DataFrame(transactions)

# Always start by looking at the shape and first few rows
print("Shape:", df_pandas.shape)  # (rows, columns)
df_pandas.head()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.2 Inspecting a DataFrame
# MAGIC
# MAGIC These are the first things you should run on any new dataset —
# MAGIC they're your "get to know you" commands.

# COMMAND ----------

# Column names and data types
print(df_pandas.dtypes)
print()

# Summary statistics for numeric columns
df_pandas.describe()

# COMMAND ----------

# Check for nulls — critical for fraud data where missing values can be suspicious
print("Null counts per column:")
print(df_pandas.isnull().sum())

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.3 Filtering and Selecting
# MAGIC
# MAGIC Two patterns you'll use constantly:
# MAGIC - **Column selection:** `df["column"]` or `df[["col1", "col2"]]`
# MAGIC - **Row filtering:** `df[df["column"] condition]`

# COMMAND ----------

# Select a single column (returns a Series)
print(df_pandas["amount"])

# COMMAND ----------

# Select multiple columns (returns a DataFrame)
df_pandas[["transaction_id", "amount", "is_fraud"]]

# COMMAND ----------

# Filter rows — show only fraudulent transactions
df_fraud = df_pandas[df_pandas["is_fraud"] == 1]
df_fraud

# COMMAND ----------

# Filter rows — high value AND fraudulent
df_pandas[(df_pandas["amount"] > 1000) & (df_pandas["is_fraud"] == 1)]

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.4 Grouping and Aggregating
# MAGIC
# MAGIC `groupby()` is one of the most important operations in data analysis.
# MAGIC Pattern: **group by a category → apply an aggregation function**
# MAGIC
# MAGIC Think of it as: "For each [X], calculate [Y]"

# COMMAND ----------

# For each country: total transaction amount and fraud count
df_pandas.groupby("country").agg(
    total_amount=("amount", "sum"),
    fraud_count=("is_fraud", "sum"),
    transaction_count=("transaction_id", "count")
).reset_index()

# COMMAND ----------

# Fraud rate by country
country_stats = df_pandas.groupby("country").agg(
    total_txns=("transaction_id", "count"),
    fraud_txns=("is_fraud", "sum")
).reset_index()

country_stats["fraud_rate"] = country_stats["fraud_txns"] / country_stats["total_txns"]
country_stats.sort_values("fraud_rate", ascending=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Part 3 — Why PySpark Exists
# MAGIC ### 3.1 The Problem With Pandas at Scale
# MAGIC
# MAGIC Pandas is excellent — but it has one fundamental limitation:
# MAGIC **everything must fit in one machine's RAM.**
# MAGIC
# MAGIC Real payment fraud datasets at a company like Revolut look like:
# MAGIC - 10+ million transactions per day
# MAGIC - Months or years of history
# MAGIC - Dozens of columns
# MAGIC
# MAGIC That's easily **hundreds of gigabytes** — far more than any single laptop can hold.
# MAGIC
# MAGIC **Apache Spark** solves this by:
# MAGIC 1. Splitting data across many machines (a "cluster")
# MAGIC 2. Running computations in parallel across all of them
# MAGIC 3. Giving you a single, clean API — you write code as if it's one machine
# MAGIC
# MAGIC **PySpark** is just the Python interface to Spark.
# MAGIC
# MAGIC ```
# MAGIC pandas:  1 machine  →  limited to ~10-50GB realistically
# MAGIC PySpark: N machines →  petabyte-scale, same code style
# MAGIC ```
# MAGIC
# MAGIC ### 3.2 The Key Mental Shift
# MAGIC
# MAGIC | pandas | PySpark |
# MAGIC |---|---|
# MAGIC | Executes immediately | Builds a plan first (lazy evaluation) |
# MAGIC | Lives in memory | Distributed across nodes |
# MAGIC | Mutable | Immutable (transformations create new DFs) |
# MAGIC | `df.head()` | `df.show()` |
# MAGIC | `df.dtypes` | `df.printSchema()` |
# MAGIC | `df.shape` | `df.count()` + `len(df.columns)` |
# MAGIC
# MAGIC The most important concept: **lazy evaluation**.
# MAGIC When you write a PySpark transformation, Spark doesn't run it immediately.
# MAGIC It builds up a **query plan** and only executes when you call an **action**
# MAGIC (like `.show()`, `.count()`, or `.write()`).
# MAGIC This is how it optimises performance under the hood.

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Part 4 — Your First PySpark DataFrame
# MAGIC ### 4.1 The SparkSession
# MAGIC
# MAGIC In Databricks, a `SparkSession` is automatically created for you and available
# MAGIC as the variable `spark`. This is your entry point to everything Spark.
# MAGIC
# MAGIC You don't need to create it — just use it.

# COMMAND ----------

# Confirm Spark is available and check the version
print(type(spark))
print("Spark version:", spark.version)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4.2 Creating a PySpark DataFrame
# MAGIC
# MAGIC We'll use the same transactions data from above.
# MAGIC The most common way to create a Spark DataFrame manually is with
# MAGIC `spark.createDataFrame()` — passing either a list of tuples + schema,
# MAGIC or a pandas DataFrame.
# MAGIC
# MAGIC We'll use the pandas → Spark conversion here so you can see the two side by side.

# COMMAND ----------

# Convert our pandas DataFrame to a Spark DataFrame
df_spark = spark.createDataFrame(df_pandas)

# In PySpark, printSchema() is your first "get to know you" command
df_spark.printSchema()

# COMMAND ----------

# show() is the PySpark equivalent of .head() in pandas
# truncate=False means don't cut off long string values
df_spark.show(truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4.3 Basic Inspection Commands
# MAGIC
# MAGIC Get comfortable with these — you'll type them dozens of times per session.

# COMMAND ----------

# Row and column count
print("Row count:", df_spark.count())
print("Column count:", len(df_spark.columns))
print("Columns:", df_spark.columns)

# COMMAND ----------

# describe() works in PySpark too — gives summary stats for numeric columns
df_spark.describe().show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4.4 Filtering in PySpark
# MAGIC
# MAGIC PySpark filtering looks very similar to pandas, but uses `.filter()` or `.where()`.
# MAGIC Both work identically — `.where()` is just an alias, kept for people coming from SQL.
# MAGIC
# MAGIC **Pattern:** `df.filter(df["column"] condition)`

# COMMAND ----------

from pyspark.sql import functions as F

# Filter for fraudulent transactions only
df_spark.filter(df_spark["is_fraud"] == 1).show(truncate=False)

# COMMAND ----------

# Filter using F.col() — this is the more idiomatic PySpark style you'll use most
# F.col("column_name") is a column reference — think of it as pointing to a column
df_spark.filter(F.col("amount") > 1000).show(truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4.5 Selecting Columns in PySpark
# MAGIC
# MAGIC Use `.select()` to choose which columns to keep (like `SELECT` in SQL).

# COMMAND ----------

# Select specific columns
df_spark.select("transaction_id", "amount", "is_fraud").show()

# COMMAND ----------

# Select with a new derived column — add a "risk_flag" column
df_spark.select(
    "transaction_id",
    "amount",
    "is_fraud",
    F.when(F.col("amount") > 1000, "HIGH").otherwise("normal").alias("risk_flag")
).show(truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## ✅ Notebook 1 Complete
# MAGIC
# MAGIC ### What you covered:
# MAGIC
# MAGIC | Topic | Key Takeaway |
# MAGIC |---|---|
# MAGIC | Python lists & dicts | Building blocks for representing rows of data |
# MAGIC | List comprehensions | Concise looping — used constantly in pipelines |
# MAGIC | Lambda functions | Anonymous one-liners, used in sorting & applying |
# MAGIC | pandas DataFrame | In-memory table — great for small data exploration |
# MAGIC | groupby / agg | "For each X, calculate Y" — your core analytics pattern |
# MAGIC | Why PySpark | Scales beyond one machine; lazy evaluation model |
# MAGIC | SparkSession | Your entry point — pre-created as `spark` in Databricks |
# MAGIC | PySpark basics | `.show()`, `.printSchema()`, `.filter()`, `.select()` |
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 🔜 What's next — Notebook 2:
# MAGIC
# MAGIC - Generate a **realistic synthetic payment fraud dataset** (10,000+ rows) using Python
# MAGIC - Save it as a **Delta table** in Databricks (your first proper data lake table)
# MAGIC - Load it back with PySpark and do your first real exploratory analysis
# MAGIC - Introduce `withColumn()`, `groupBy()`, `orderBy()` on real data