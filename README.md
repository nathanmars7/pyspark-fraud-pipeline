# PySpark Fraud Pipeline

A payment fraud detection pipeline built with PySpark and Delta Lake on Databricks, delivered as five notebooks. It generates a synthetic transaction dataset, engineers fraud signals with window functions, builds bronze, silver and gold Delta tables, and finishes with a reusable data quality checker, drift detection and Delta constraints.

## At a glance

| | |
|---|---|
| Stack | PySpark, Delta Lake, Spark SQL, pandas and numpy (data generation only) |
| Platform | Databricks notebooks; all tables live in `workspace.default` |
| Data | 10,000 synthetic transactions generated in notebook 02 (seeded, 2024 calendar year, 500 users, 200 merchants, ~5% fraud rate) |
| What it shows | Medallion layering, time-range window functions, rule-based risk scoring, left and anti joins, `.transform()` chains, a chainable `DataQualityChecker` class, fraud rate drift flags, `ALTER TABLE ... ADD CONSTRAINT` |
| Status | Portfolio project, complete. Last verified 2026-05 from the repo contents. |

## Architecture

```
Bronze                Silver                 Gold                     Monitoring
──────                ──────                 ────                     ──────────
fraud_transactions -> fraud_features    ->   fraud_user_profile       fraud_dq_report
(raw transactions)    (window features) ->   fraud_merchant_stats
                                        ->   fraud_gold_enriched
```

Six Delta tables in total. Each notebook pins `USE CATALOG workspace` and `USE SCHEMA default`, and writes with `saveAsTable()` using the full three-part name. Tables are dropped and recreated on each run.

## Notebooks

| # | Notebook | What it does |
|---|---|---|
| 01 | `nb01_payment_fraud_setup_and_python_refresher.py` | Python and pandas refresher on a five-row toy dataset, then the same data as a Spark DataFrame: `printSchema()`, `show()`, `filter()`, `select()`, `F.when()`. Writes no tables. |
| 02 | `nb02_synthetic_data_and_delta_table.py` | Generates 10,000 transactions with weighted fraud patterns (country, payment method, amount, night-time hours), writes `fraud_transactions` as a Delta table, shows `DESCRIBE HISTORY` and `versionAsOf` time travel, then explores it with `withColumn()`, `groupBy().agg()` and `%sql` cells. |
| 03 | `nb03_window_functions_and_feature_engineering.py` | Builds velocity features (`rangeBetween` over 1h, 24h, 7d), behavioural features (running average, amount ratio, first transaction), compares `row_number()`, `rank()` and `dense_rank()`, and combines seven signals into a `risk_score` (max 11) with a `LOW / MEDIUM / HIGH` label. Writes `fraud_features`. |
| 04 | `nb04_aggregations_joins_and_gold_layer.py` | Aggregates per-user and per-merchant profiles with risk tiers, demonstrates left and anti joins, wraps enrichments in `.transform()` functions and runs them as one `run_gold_pipeline()` call. Writes `fraud_user_profile`, `fraud_merchant_stats`, `fraud_gold_enriched`. |
| 05 | `nb05_data_quality_and_monitoring.py` | Defines a chainable `DataQualityChecker` (row count, nulls, uniqueness, value range, allowed values, metric range) and runs it on four tables, flags months whose fraud rate is outside 50% to 150% of the overall rate, adds two Delta constraints to `fraud_transactions` and proves a bad append is rejected. Writes `fraud_dq_report`. |

## Run it

1. In a Databricks workspace, create a folder and import the five `.py` files (Workspace, Import, File). They are in Databricks notebook source format and import as notebooks.
2. Attach a cluster running a Databricks Runtime that includes Delta Lake and PySpark (any current runtime does). The notebooks use the `workspace` catalog and `default` schema, so the cluster needs access to both.
3. Run the notebooks in order, 01 to 05. Notebooks 03 to 05 read the tables written by the ones before them.

Requirements: no `requirements.txt` is needed. The notebooks import only `pyspark`, `pandas`, `numpy` and the standard library, all of which Databricks Runtime provides.

## Limitations

- Synthetic data: fraud patterns are written into the generator, not learned from real transactions.
- No ML model: the risk score is rule-based. There is no orchestration and no streaming; everything is batch run by hand.
- `fraud_merchant_stats` is keyed on `(merchant_id, merchant_category)`, so the uniqueness check on `merchant_id` alone in notebook 05 reports duplicates by design.

## Portfolio

Part of a set of data engineering portfolio projects:

- [dbt-analytics-engineering](https://github.com/nathanmars7/dbt-analytics-engineering): dbt on DuckDB and Snowflake from one codebase
- [jaffle-shop-snowflake](https://github.com/nathanmars7/jaffle-shop-snowflake): minimal dbt starter on Snowflake
- [dbt-nyc-citibike](https://github.com/nathanmars7/dbt-nyc-citibike): dbt on BigQuery, Citibike ridership vs weather
- [dbt-crypto-bitcoin](https://github.com/nathanmars7/dbt-crypto-bitcoin): dbt on BigQuery with snapshots, Bitcoin transactions
- [dbt-learning](https://github.com/nathanmars7/dbt-learning): dbt on BigQuery, Chicago taxi trips, incremental models
- [pyspark-fraud-pipeline](https://github.com/nathanmars7/pyspark-fraud-pipeline) (this repo): PySpark and Delta Lake medallion pipeline on Databricks
