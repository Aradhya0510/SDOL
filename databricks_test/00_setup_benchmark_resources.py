# Databricks notebook source
# MAGIC %md
# MAGIC # SDOL Benchmark: Setup Resources
# MAGIC
# MAGIC This notebook creates synthetic OLAP and OLTP datasets in Unity Catalog for benchmarking
# MAGIC a **baseline Databricks MCP agent** against an **SDOL-enhanced agent**.
# MAGIC
# MAGIC | Table | Paradigm | Rows | Description |
# MAGIC |-------|----------|------|-------------|
# MAGIC | `customers` | OLTP | 1,000 | Customer profiles — tier, region, lifetime value |
# MAGIC | `products` | OLTP | 200 | Product catalog — category, price, stock |
# MAGIC | `orders` | OLTP | 5,000 | Recent orders — status, amounts, timestamps |
# MAGIC | `sales_transactions` | OLAP | 100,000 | Historical sales fact table |
# MAGIC | `revenue_daily` | OLAP | ~1,460 | Pre-aggregated daily revenue by region |
# MAGIC
# MAGIC **Prerequisites:** Unity Catalog enabled workspace with CREATE CATALOG / CREATE SCHEMA permissions.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration
# MAGIC Update `CATALOG` and `SCHEMA` for your workspace. Set `USE_EXISTING_CATALOG = True` if the catalog already exists.

# COMMAND ----------

CATALOG = "users"  # TODO: change if needed
SCHEMA = "aradhya_chouhan"
USE_EXISTING_CATALOG = False

# COMMAND ----------

if not USE_EXISTING_CATALOG:
    spark.sql(f"CREATE CATALOG IF NOT EXISTS {CATALOG}")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")
spark.sql(f"USE CATALOG {CATALOG}")
spark.sql(f"USE SCHEMA {SCHEMA}")
print(f"✔ Using {CATALOG}.{SCHEMA}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Shared Constants

# COMMAND ----------

from pyspark.sql import functions as F

FIRST_NAMES = [
    "Alice", "Bob", "Charlie", "Diana", "Eve", "Frank", "Grace", "Henry",
    "Iris", "Jack", "Kate", "Leo", "Maya", "Noah", "Olivia", "Pete",
    "Quinn", "Rose", "Sam", "Tara", "Uma", "Victor", "Wendy", "Xander",
    "Yuki", "Zara", "Adam", "Beth", "Carl", "Dana",
]
LAST_NAMES = [
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller",
    "Davis", "Wilson", "Moore", "Taylor", "Anderson", "Thomas", "Jackson",
    "White", "Harris", "Martin", "Thompson", "Clark", "Lewis", "Walker",
    "Hall", "Allen", "Young", "King", "Wright", "Hill", "Scott", "Green", "Baker",
]
TIERS = ["free", "pro", "enterprise"]
REGIONS = ["west", "east", "central", "south"]
CATEGORIES = ["Electronics", "Clothing", "Home & Garden", "Sports", "Books"]
CHANNELS = ["online", "in_store", "mobile"]
SUPPLIERS = ["Apex Corp", "BlueLine Inc", "CedarTech", "Delta Supply", "EverGoods"]
ORDER_STATUSES = ["completed", "pending", "cancelled", "refunded"]

first_arr = F.array(*[F.lit(n) for n in FIRST_NAMES])
last_arr = F.array(*[F.lit(n) for n in LAST_NAMES])
region_arr = F.array(*[F.lit(r) for r in REGIONS])
category_arr = F.array(*[F.lit(c) for c in CATEGORIES])
channel_arr = F.array(*[F.lit(c) for c in CHANNELS])
supplier_arr = F.array(*[F.lit(s) for s in SUPPLIERS])

def pick(arr, n, seed):
    """Random 1-based index into a PySpark array literal."""
    return F.element_at(arr, F.floor(F.rand(seed) * n).cast("int") + 1)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1 — Customers (OLTP · 1,000 rows)

# COMMAND ----------

customers_df = (
    spark.range(1, 1001)
    .withColumn("customer_id", F.format_string("C-%04d", "id"))
    .withColumn("name", F.concat_ws(" ", pick(first_arr, 30, 42), pick(last_arr, 30, 43)))
    .withColumn("email", F.concat(
        F.lower(pick(first_arr, 30, 42)), F.lit("."),
        F.lower(pick(last_arr, 30, 43)),
        F.format_string("%d", F.col("id")),
        F.lit("@acme.com"),
    ))
    .withColumn("_tr", F.rand(44))
    .withColumn("tier",
        F.when(F.col("_tr") < 0.60, F.lit("free"))
         .when(F.col("_tr") < 0.90, F.lit("pro"))
         .otherwise(F.lit("enterprise")),
    )
    .withColumn("region", pick(region_arr, 4, 45))
    .withColumn("signup_date", F.date_add(F.lit("2024-01-01"), (F.rand(46) * 450).cast("int")))
    .withColumn("last_login", F.from_unixtime(
        F.unix_timestamp(F.current_timestamp()) - (F.rand(47) * 30 * 86400).cast("long")
    ))
    .withColumn("lifetime_value", F.round(
        F.when(F.col("tier") == "enterprise", F.rand(48) * 50000 + 10000)
         .when(F.col("tier") == "pro", F.rand(48) * 10000 + 1000)
         .otherwise(F.rand(48) * 500 + 50), 2,
    ))
    .withColumn("is_active", F.rand(49) > 0.08)
    .drop("id", "_tr")
)

customers_df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(f"{CATALOG}.{SCHEMA}.customers")
print(f"✔ customers: {spark.table(f'{CATALOG}.{SCHEMA}.customers').count()} rows")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2 — Products (OLTP · 200 rows)

# COMMAND ----------

products_df = (
    spark.range(1, 201)
    .withColumn("product_id", F.format_string("P-%03d", "id"))
    .withColumn("category", pick(category_arr, 5, 50))
    .withColumn("name", F.concat(F.col("category"), F.lit(" Item "), F.format_string("%03d", "id")))
    .withColumn("price", F.round(
        F.when(F.col("category") == "Electronics", F.rand(51) * 500 + 49.99)
         .when(F.col("category") == "Clothing",    F.rand(51) * 150 + 19.99)
         .when(F.col("category") == "Home & Garden", F.rand(51) * 300 + 29.99)
         .when(F.col("category") == "Sports",      F.rand(51) * 200 + 24.99)
         .otherwise(F.rand(51) * 50 + 4.99), 2,
    ))
    .withColumn("stock_quantity", (F.rand(52) * 500).cast("int"))
    .withColumn("supplier", pick(supplier_arr, 5, 53))
    .withColumn("is_available", F.rand(54) > 0.05)
    .drop("id")
)

products_df.write.mode("overwrite").saveAsTable(f"{CATALOG}.{SCHEMA}.products")
print(f"✔ products: {spark.table(f'{CATALOG}.{SCHEMA}.products').count()} rows")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3 — Orders (OLTP · 5,000 rows)

# COMMAND ----------

orders_df = (
    spark.range(1, 5001)
    .withColumn("order_id", F.format_string("ORD-%06d", "id"))
    .withColumn("customer_id", F.format_string("C-%04d", F.floor(F.rand(55) * 1000).cast("int") + 1))
    .withColumn("product_id", F.format_string("P-%03d", F.floor(F.rand(56) * 200).cast("int") + 1))
    .withColumn("quantity", F.floor(F.rand(57) * 9).cast("int") + 1)
    .withColumn("unit_price", F.round(F.rand(61) * 200 + 10, 2))
    .withColumn("total_amount", F.round(F.col("quantity") * F.col("unit_price"), 2))
    .withColumn("_sr", F.rand(58))
    .withColumn("status",
        F.when(F.col("_sr") < 0.75, F.lit("completed"))
         .when(F.col("_sr") < 0.88, F.lit("pending"))
         .when(F.col("_sr") < 0.95, F.lit("cancelled"))
         .otherwise(F.lit("refunded")),
    )
    .withColumn("created_at", F.from_unixtime(
        F.unix_timestamp(F.current_timestamp()) - (F.rand(59) * 90 * 86400).cast("long")
    ))
    .withColumn("updated_at", F.from_unixtime(
        F.unix_timestamp(F.col("created_at")) + (F.rand(60) * 3 * 86400).cast("long")
    ))
    .drop("id", "_sr")
)

orders_df.write.mode("overwrite").saveAsTable(f"{CATALOG}.{SCHEMA}.orders")
print(f"✔ orders: {spark.table(f'{CATALOG}.{SCHEMA}.orders').count()} rows")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4 — Sales Transactions (OLAP · 100,000 rows)
# MAGIC Large fact table for analytical queries — aggregations, trends, top-N.

# COMMAND ----------

sales_df = (
    spark.range(1, 100001)
    .withColumn("transaction_id", F.col("id"))
    .withColumn("customer_id", F.format_string("C-%04d", F.floor(F.rand(62) * 1000).cast("int") + 1))
    .withColumn("product_id", F.format_string("P-%03d", F.floor(F.rand(63) * 200).cast("int") + 1))
    .withColumn("quantity", F.floor(F.rand(64) * 8).cast("int") + 1)
    .withColumn("unit_price", F.round(F.rand(65) * 200 + 10, 2))
    .withColumn("total_amount", F.round(F.col("quantity") * F.col("unit_price"), 2))
    .withColumn("region", pick(region_arr, 4, 66))
    .withColumn("channel", pick(channel_arr, 3, 67))
    .withColumn("order_date", F.date_add(F.lit("2025-04-01"), (F.rand(68) * 365).cast("int")))
    .withColumn("_sr", F.rand(69))
    .withColumn("status",
        F.when(F.col("_sr") < 0.80, F.lit("completed"))
         .when(F.col("_sr") < 0.90, F.lit("pending"))
         .when(F.col("_sr") < 0.95, F.lit("cancelled"))
         .otherwise(F.lit("refunded")),
    )
    .drop("id", "_sr")
)

sales_df.write.mode("overwrite").saveAsTable(f"{CATALOG}.{SCHEMA}.sales_transactions")
print(f"✔ sales_transactions: {spark.table(f'{CATALOG}.{SCHEMA}.sales_transactions').count()} rows")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5 — Revenue Daily (OLAP · ~1,460 rows)
# MAGIC Pre-aggregated from `sales_transactions` (completed only), grouped by date × region.

# COMMAND ----------

revenue_daily_df = (
    spark.table(f"{CATALOG}.{SCHEMA}.sales_transactions")
    .filter(F.col("status") == "completed")
    .groupBy(
        F.col("order_date").alias("report_date"),
        "region",
    )
    .agg(
        F.round(F.sum("total_amount"), 2).alias("total_revenue"),
        F.count("*").alias("order_count"),
        F.round(F.avg("total_amount"), 2).alias("avg_order_value"),
    )
    .orderBy("report_date", "region")
)

revenue_daily_df.write.mode("overwrite").saveAsTable(f"{CATALOG}.{SCHEMA}.revenue_daily")
print(f"✔ revenue_daily: {spark.table(f'{CATALOG}.{SCHEMA}.revenue_daily').count()} rows")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Verification

# COMMAND ----------

from pyspark.sql import Row

tables = ["customers", "products", "orders", "sales_transactions", "revenue_daily"]
summary = [Row(table=t, rows=spark.table(f"{CATALOG}.{SCHEMA}.{t}").count()) for t in tables]
display(spark.createDataFrame(summary))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Sample Data

# COMMAND ----------

display(spark.sql(f"SELECT * FROM {CATALOG}.{SCHEMA}.customers WHERE customer_id IN ('C-0042', 'C-0100', 'C-0500') ORDER BY customer_id"))

# COMMAND ----------

display(spark.sql(f"SELECT * FROM {CATALOG}.{SCHEMA}.products WHERE product_id IN ('P-001', 'P-105', 'P-200') ORDER BY product_id"))

# COMMAND ----------

display(spark.sql(f"""
  SELECT region, COUNT(*) AS txn_count, ROUND(SUM(total_amount), 2) AS total_rev
  FROM {CATALOG}.{SCHEMA}.sales_transactions
  WHERE status = 'completed'
  GROUP BY region ORDER BY total_rev DESC
"""))

# COMMAND ----------

display(spark.sql(f"SELECT * FROM {CATALOG}.{SCHEMA}.revenue_daily ORDER BY report_date DESC LIMIT 20"))

# COMMAND ----------

# MAGIC %md
# MAGIC **Setup complete.** Proceed to `01_baseline_vs_sdol_benchmark` to run the evaluation.
