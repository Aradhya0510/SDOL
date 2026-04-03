# Databricks notebook source
# MAGIC %md
# MAGIC # Fleet Management: Data Setup
# MAGIC
# MAGIC Creates the synthetic data landscape for the Provena fleet-management showcase.
# MAGIC The scenario models a heavy-machinery manufacturer with excavators, generators,
# MAGIC and industrial equipment deployed globally.
# MAGIC
# MAGIC | Table | Paradigm | Rows | Description |
# MAGIC |-------|----------|------|-------------|
# MAGIC | `fleet_machines` | OLTP | 500 | Real-time machine registry — status, firmware, GPS |
# MAGIC | `telemetry_readings` | OLAP | ~360K | Hourly sensor data — temp, RPM, fuel efficiency |
# MAGIC | `telemetry_daily` | OLAP | ~90K | Pre-aggregated daily metrics (15-min batch lag) |
# MAGIC | `maintenance_logs` | Document | ~5,000 | Free-text technician notes + fault categories |
# MAGIC
# MAGIC **Conflict seed:** Machine `EXC-0342` is `offline` in the OLTP registry but still
# MAGIC shows `online` in the OLAP daily aggregate (batch hasn't caught up yet). This
# MAGIC engineered discrepancy lets the benchmark demonstrate Provena's conflict detection.
# MAGIC
# MAGIC **Prerequisites:** Unity Catalog workspace. A Databricks Vector Search endpoint
# MAGIC (the notebook creates one if needed).

# COMMAND ----------

# MAGIC %pip install -U -qqqq pydantic>=2.0 databricks-vectorsearch

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

dbutils.widgets.text("catalog", "users")
dbutils.widgets.text("schema", "default")
dbutils.widgets.text("vs_endpoint", "provena_fleet_vs")
dbutils.widgets.text("embedding_model", "databricks-bge-large-en")
dbutils.widgets.dropdown("use_existing_catalog", "true", ["true", "false"])

CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")
USE_EXISTING_CATALOG = dbutils.widgets.get("use_existing_catalog") == "true"

VS_ENDPOINT_NAME = dbutils.widgets.get("vs_endpoint")
VS_INDEX_NAME = f"{CATALOG}.{SCHEMA}.maintenance_logs_index"
EMBEDDING_MODEL = dbutils.widgets.get("embedding_model")

# COMMAND ----------

if not USE_EXISTING_CATALOG:
    spark.sql(f"CREATE CATALOG IF NOT EXISTS {CATALOG}")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")
spark.sql(f"USE CATALOG {CATALOG}")
spark.sql(f"USE SCHEMA {SCHEMA}")
print(f"Using {CATALOG}.{SCHEMA}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Shared Constants

# COMMAND ----------

from pyspark.sql import functions as F
import random

MODELS = ["Model X", "Model Y", "Model Z"]
FIRMWARE_VERSIONS = ["v1.9", "v2.0", "v2.1", "v2.2"]
STATUSES = ["online", "offline", "maintenance"]
REGIONS = ["north_america", "europe", "asia_pacific", "middle_east"]
FAULT_CATEGORIES = ["overheating", "fuel_system", "vibration", "electrical", "hydraulic", "routine"]
SEVERITIES = ["low", "medium", "high", "critical"]
TECHNICIANS = [
    "Sarah Chen", "Marcus Rivera", "Aisha Patel", "Dmitri Volkov",
    "Elena Kowalski", "James Okonkwo", "Yuki Tanaka", "Carlos Mendez",
    "Fatima Al-Hassan", "Liam O'Brien",
]

model_arr = F.array(*[F.lit(m) for m in MODELS])
fw_arr = F.array(*[F.lit(v) for v in FIRMWARE_VERSIONS])
status_arr = F.array(*[F.lit(s) for s in STATUSES])
region_arr = F.array(*[F.lit(r) for r in REGIONS])

def pick(arr, n, seed):
    return F.element_at(arr, F.floor(F.rand(seed) * n).cast("int") + 1)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1 — Fleet Machines (OLTP · 500 rows)
# MAGIC
# MAGIC Real-time machine registry. `EXC-0342` is explicitly set to `offline` to seed
# MAGIC the epistemic conflict with the OLAP telemetry data.

# COMMAND ----------

machines_df = (
    spark.range(1, 501)
    .withColumn("machine_id", F.format_string("EXC-%04d", "id"))
    .withColumn("model", pick(model_arr, 3, 70))
    .withColumn("serial_number", F.concat(F.lit("SN-"), F.format_string("%06d", F.col("id") * 7 + 1000)))
    .withColumn("firmware_version", pick(fw_arr, 4, 71))
    .withColumn("_sr", F.rand(72))
    .withColumn("status",
        F.when(F.col("_sr") < 0.75, F.lit("online"))
         .when(F.col("_sr") < 0.90, F.lit("offline"))
         .otherwise(F.lit("maintenance")),
    )
    .withColumn("region", pick(region_arr, 4, 73))
    .withColumn("gps_lat", F.round(F.rand(74) * 120 - 60, 4))
    .withColumn("gps_lon", F.round(F.rand(75) * 360 - 180, 4))
    .withColumn("last_heartbeat_at", F.from_unixtime(
        F.unix_timestamp(F.current_timestamp()) - (F.rand(76) * 300).cast("long")
    ))
    .drop("id", "_sr")
)

machines_df = machines_df.withColumn(
    "status",
    F.when(F.col("machine_id") == "EXC-0342", F.lit("offline")).otherwise(F.col("status")),
).withColumn(
    "model",
    F.when(F.col("machine_id") == "EXC-0342", F.lit("Model X")).otherwise(F.col("model")),
).withColumn(
    "firmware_version",
    F.when(F.col("machine_id") == "EXC-0342", F.lit("v2.1")).otherwise(F.col("firmware_version")),
).withColumn(
    "last_heartbeat_at",
    F.when(F.col("machine_id") == "EXC-0342", F.current_timestamp()).otherwise(F.col("last_heartbeat_at")),
)

machines_df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(f"{CATALOG}.{SCHEMA}.fleet_machines")
print(f"fleet_machines: {spark.table(f'{CATALOG}.{SCHEMA}.fleet_machines').count()} rows")

# COMMAND ----------

display(spark.sql(f"SELECT * FROM {CATALOG}.{SCHEMA}.fleet_machines WHERE machine_id = 'EXC-0342'"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2 — Telemetry Readings (OLAP · ~360K rows)
# MAGIC
# MAGIC Hourly sensor data for the last 180 days across all 500 machines.
# MAGIC Model X machines on firmware v2.1 get slightly worse fuel efficiency to make
# MAGIC the cross-paradigm query interesting.

# COMMAND ----------

machines_meta = spark.table(f"{CATALOG}.{SCHEMA}.fleet_machines").select("machine_id", "model", "firmware_version")

readings_base = (
    spark.range(0, 500 * 180 * 4)
    .withColumn("machine_idx", (F.col("id") % 500).cast("int") + 1)
    .withColumn("day_offset", (F.floor(F.col("id") / 500 / 4)).cast("int"))
    .withColumn("hour_slot", ((F.col("id") / 500).cast("int") % 4) * 6)
    .withColumn("machine_id", F.format_string("EXC-%04d", "machine_idx"))
    .withColumn("reading_time", F.expr(
        f"timestamp(date_add(current_date() - 180, day_offset)) + make_interval(0,0,0,0,hour_slot,0,0)"
    ))
    .drop("machine_idx", "day_offset", "hour_slot")
)

readings_df = (
    readings_base
    .join(machines_meta, "machine_id")
    .withColumn("_is_degraded",
        (F.col("model") == "Model X") & (F.col("firmware_version") == "v2.1")
    )
    .withColumn("engine_temp_c", F.round(
        F.when(F.col("_is_degraded"), F.rand(80) * 40 + 85)
         .otherwise(F.rand(80) * 30 + 60), 1
    ))
    .withColumn("rpm", (
        F.when(F.col("_is_degraded"), F.rand(81) * 800 + 1800)
         .otherwise(F.rand(81) * 600 + 1200)
    ).cast("int"))
    .withColumn("fuel_efficiency_lpkm", F.round(
        F.when(F.col("_is_degraded"), F.rand(82) * 4 + 14)
         .otherwise(F.rand(82) * 3 + 8), 2
    ))
    .withColumn("vibration_mm_s", F.round(
        F.when(F.col("_is_degraded"), F.rand(83) * 3 + 2.5)
         .otherwise(F.rand(83) * 2 + 0.5), 2
    ))
    .withColumn("oil_pressure_psi", F.round(F.rand(84) * 20 + 40, 1))
    .withColumn("coolant_temp_c", F.round(F.rand(85) * 15 + 75, 1))
    .drop("id", "model", "firmware_version", "_is_degraded")
)

readings_df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(f"{CATALOG}.{SCHEMA}.telemetry_readings")
cnt = spark.table(f"{CATALOG}.{SCHEMA}.telemetry_readings").count()
print(f"telemetry_readings: {cnt} rows")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3 — Telemetry Daily (OLAP · ~90K rows)
# MAGIC
# MAGIC Pre-aggregated from `telemetry_readings`. The `last_known_status` column
# MAGIC is derived from whether the machine had recent readings — simulating a
# MAGIC 15-minute batch-update lag. For `EXC-0342`, this will be `'online'` even
# MAGIC though the real-time OLTP registry says `'offline'`.

# COMMAND ----------

telemetry_daily_df = (
    spark.table(f"{CATALOG}.{SCHEMA}.telemetry_readings")
    .groupBy(
        "machine_id",
        F.to_date("reading_time").alias("report_date"),
    )
    .agg(
        F.round(F.avg("engine_temp_c"), 1).alias("avg_engine_temp"),
        F.round(F.max("engine_temp_c"), 1).alias("max_engine_temp"),
        F.round(F.avg("rpm"), 0).alias("avg_rpm"),
        F.round(F.avg("fuel_efficiency_lpkm"), 2).alias("avg_fuel_efficiency"),
        F.round(F.min("fuel_efficiency_lpkm"), 2).alias("min_fuel_efficiency"),
        F.count("*").alias("reading_count"),
    )
    .withColumn("last_known_status", F.lit("online"))
    .orderBy("machine_id", "report_date")
)

telemetry_daily_df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(f"{CATALOG}.{SCHEMA}.telemetry_daily")

# --- Conflict seed: insert a synthetic "today" row for EXC-0342 ---
# The OLAP daily table will show 'online' for today, while OLTP says 'offline'.
# Without this row, the OLAP query for today returns zero rows and no conflict fires.
from pyspark.sql import Row
conflict_row = spark.createDataFrame([Row(
    machine_id="EXC-0342",
    report_date=spark.sql("SELECT current_date()").first()[0],
    avg_engine_temp=105.3,
    max_engine_temp=125.0,
    avg_rpm=2100.0,
    avg_fuel_efficiency=16.2,
    min_fuel_efficiency=14.8,
    reading_count=4,
    last_known_status="online",  # contradicts OLTP 'offline'
)])
conflict_row.write.mode("append").saveAsTable(f"{CATALOG}.{SCHEMA}.telemetry_daily")

cnt = spark.table(f"{CATALOG}.{SCHEMA}.telemetry_daily").count()
print(f"telemetry_daily: {cnt} rows (includes conflict seed for today)")

# COMMAND ----------

display(spark.sql(f"""
  SELECT * FROM {CATALOG}.{SCHEMA}.telemetry_daily
  WHERE machine_id = 'EXC-0342' ORDER BY report_date DESC LIMIT 5
"""))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4 — Maintenance Logs (Document · ~5,000 rows)
# MAGIC
# MAGIC Realistic free-text technician notes. Model X + v2.1 machines get more
# MAGIC overheating / fuel-system faults to make the semantic search showcase work.

# COMMAND ----------

OVERHEATING_TEMPLATES = [
    "Engine temperature exceeded {temp}°C during sustained operation. Coolant levels nominal. Suspect radiator blockage. Cleaned debris from intake vents and monitored for 2 hours — temperature stabilized at {norm_temp}°C.",
    "Thermal shutdown triggered at {temp}°C. Ambient conditions extreme ({ambient}°C). Replaced thermal paste on heat exchanger. Post-repair readings within spec.",
    "Recurring high-temperature alerts. Engine reaching {temp}°C under moderate load. Inspected thermostat — found stuck in partially closed position. Replaced thermostat assembly.",
    "Overheating event during heavy excavation work. Peak temperature {temp}°C. Root cause: clogged oil cooler fins. Cleaned and flushed cooling system. Recommended firmware update for improved fan control.",
]
FUEL_TEMPLATES = [
    "Fuel consumption {pct}% above baseline for this model. Injector spray pattern analysis shows uneven distribution on cylinder 3. Replaced fuel injector set. Efficiency improved from {bad_eff} to {good_eff} L/km.",
    "Operator reported excessive fuel usage during idle. Diagnosed faulty fuel pressure regulator allowing bypass flow. Replaced regulator and recalibrated ECU. Fuel efficiency normalized.",
    "Fuel system diagnostic: high-pressure pump showing intermittent pressure drops. Fuel filter heavily contaminated. Replaced filter and pump assembly. Ran 4-hour load test — consumption within {good_eff} L/km spec.",
    "Abnormal fuel efficiency degradation over past month ({bad_eff} L/km vs {good_eff} L/km expected). Found air leak in intake manifold gasket. Replaced gasket and reset adaptive fuel trim.",
]
VIBRATION_TEMPLATES = [
    "Excessive vibration detected at {vib} mm/s (threshold: 3.0 mm/s). Spectrum analysis indicates bearing wear on main drive shaft. Replaced bearings and realigned shaft. Post-repair vibration: {norm_vib} mm/s.",
    "Operator complaint: unusual shaking during operation. Vibration sensor confirmed {vib} mm/s at operating RPM. Found loose mounting bolts on engine cradle. Torqued to spec, vibration resolved.",
]
ELECTRICAL_TEMPLATES = [
    "Intermittent sensor dropouts on CAN bus. Diagnosed corroded connector on harness J4. Replaced connector and applied dielectric grease. All sensors reporting normally after 24-hour soak test.",
    "Battery voltage fluctuations causing ECU resets. Found alternator belt slippage. Replaced belt and tensioner. Charging system output stable at 28.2V.",
]
HYDRAULIC_TEMPLATES = [
    "Hydraulic cylinder response sluggish. Measured {psi} PSI (expected 3000 PSI). Found internal leak in control valve. Rebuilt valve assembly and replaced seals. System pressure restored.",
    "Hydraulic fluid contamination detected during routine sample. Particle count exceeded ISO 18/15 limits. Performed full fluid flush and filter replacement. Resampled within spec.",
]
ROUTINE_TEMPLATES = [
    "Scheduled 500-hour service completed. Oil and filter change, air filter replacement, belt inspection. All systems nominal. Next service due at {next_hrs} hours.",
    "Annual inspection: all safety systems functional. Structural integrity check passed. Updated firmware from {old_fw} to {new_fw}. Calibrated GPS and telematics module.",
]

import random as _rng

def _gen_logs(spark_session, n_logs=5000):
    _rng.seed(42)
    rows = []
    cats_templates = {
        "overheating": OVERHEATING_TEMPLATES,
        "fuel_system": FUEL_TEMPLATES,
        "vibration": VIBRATION_TEMPLATES,
        "electrical": ELECTRICAL_TEMPLATES,
        "hydraulic": HYDRAULIC_TEMPLATES,
        "routine": ROUTINE_TEMPLATES,
    }
    machines_info = {
        r.machine_id: (r.model, r.firmware_version)
        for r in spark_session.table(f"{CATALOG}.{SCHEMA}.fleet_machines")
            .select("machine_id", "model", "firmware_version").collect()
    }
    machine_ids = list(machines_info.keys())
    for i in range(1, n_logs + 1):
        mid = _rng.choice(machine_ids)
        model, fw = machines_info[mid]
        if model == "Model X" and fw == "v2.1" and _rng.random() < 0.6:
            cat = _rng.choice(["overheating", "fuel_system"])
        else:
            cat = _rng.choice(list(cats_templates.keys()))
        sev = _rng.choice(SEVERITIES)
        if cat in ("overheating", "fuel_system") and model == "Model X":
            sev = _rng.choice(["high", "critical", "high", "medium"])
        tmpl = _rng.choice(cats_templates[cat])
        desc = tmpl.format(
            temp=_rng.randint(105, 130), norm_temp=_rng.randint(80, 95),
            ambient=_rng.randint(38, 50), pct=_rng.randint(15, 40),
            bad_eff=round(_rng.uniform(14, 18), 1),
            good_eff=round(_rng.uniform(9, 12), 1),
            vib=round(_rng.uniform(3.5, 6.0), 1),
            norm_vib=round(_rng.uniform(0.5, 1.5), 1),
            psi=_rng.randint(1800, 2400),
            next_hrs=_rng.randint(4500, 6000),
            old_fw=_rng.choice(["v1.9", "v2.0"]),
            new_fw=_rng.choice(["v2.1", "v2.2"]),
        )
        tech = _rng.choice(TECHNICIANS)
        log_date = f"2025-{_rng.randint(1, 12):02d}-{_rng.randint(1, 28):02d}"
        rows.append((
            f"LOG-{i:06d}", mid, log_date, cat, sev, desc, tech,
            f"Issue resolved. Parts on order." if sev in ("high", "critical") else "Resolved on site.",
        ))
    schema = "log_id STRING, machine_id STRING, log_date STRING, fault_category STRING, severity STRING, description STRING, technician_name STRING, resolution_notes STRING"
    return spark_session.createDataFrame(rows, schema=schema)

logs_df = _gen_logs(spark)
logs_df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(f"{CATALOG}.{SCHEMA}.maintenance_logs")
cnt = spark.table(f"{CATALOG}.{SCHEMA}.maintenance_logs").count()
print(f"maintenance_logs: {cnt} rows")

# COMMAND ----------

display(spark.sql(f"""
  SELECT fault_category, COUNT(*) AS cnt
  FROM {CATALOG}.{SCHEMA}.maintenance_logs
  WHERE machine_id IN (
    SELECT machine_id FROM {CATALOG}.{SCHEMA}.fleet_machines
    WHERE model = 'Model X' AND firmware_version = 'v2.1'
  )
  GROUP BY fault_category ORDER BY cnt DESC
"""))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5 — Databricks Vector Search Index
# MAGIC
# MAGIC Creates a VS endpoint (if needed) and a delta-sync index on `maintenance_logs.description`
# MAGIC so the Provena benchmark can perform real semantic search.

# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient

vsc = VectorSearchClient()

existing_endpoints = [ep["name"] for ep in vsc.list_endpoints().get("endpoints", [])]
if VS_ENDPOINT_NAME not in existing_endpoints:
    print(f"Creating Vector Search endpoint '{VS_ENDPOINT_NAME}'...")
    vsc.create_endpoint(name=VS_ENDPOINT_NAME, endpoint_type="STANDARD")
    print("Endpoint created (may take a few minutes to become ready).")
else:
    print(f"Endpoint '{VS_ENDPOINT_NAME}' already exists.")

# COMMAND ----------

existing_indexes = [
    idx["name"]
    for idx in vsc.list_indexes(name=VS_ENDPOINT_NAME).get("vector_indexes", [])
]

source_table = f"{CATALOG}.{SCHEMA}.maintenance_logs"

spark.sql(f"ALTER TABLE {source_table} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")

if VS_INDEX_NAME not in existing_indexes:
    print(f"Creating delta-sync index '{VS_INDEX_NAME}'...")
    try:
        vsc.create_delta_sync_index(
            endpoint_name=VS_ENDPOINT_NAME,
            index_name=VS_INDEX_NAME,
            source_table_name=source_table,
            pipeline_type="TRIGGERED",
            primary_key="log_id",
            embedding_source_column="description",
            embedding_model_endpoint_name=EMBEDDING_MODEL,
            columns_to_sync=["log_id", "machine_id", "log_date", "fault_category", "severity", "description", "technician_name"],
        )
        print("Index creation started. Waiting for initial sync...")
    except Exception as e:
        if "already exists" in str(e):
            print(f"Index already exists (race condition). Triggering sync...")
            vsc.get_index(endpoint_name=VS_ENDPOINT_NAME, index_name=VS_INDEX_NAME).sync()
        else:
            raise
else:
    print(f"Index '{VS_INDEX_NAME}' already exists. Triggering sync...")
    vsc.get_index(endpoint_name=VS_ENDPOINT_NAME, index_name=VS_INDEX_NAME).sync()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Wait for index to be ready

# COMMAND ----------

import time

idx = vsc.get_index(endpoint_name=VS_ENDPOINT_NAME, index_name=VS_INDEX_NAME)
for attempt in range(60):
    status = idx.describe()
    state = status.get("status", {}).get("detailed_state", "UNKNOWN")
    if state == "ONLINE_NO_PENDING_UPDATE":
        print(f"Index is ONLINE and ready.")
        break
    print(f"  [{attempt+1}/60] Index state: {state} — waiting 30s...")
    time.sleep(30)
else:
    print("WARNING: Index did not become ready within 30 minutes. Check the VS UI.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Verify Vector Search

# COMMAND ----------

results = idx.similarity_search(
    query_text="Model X overheating failure high temperature",
    columns=["log_id", "machine_id", "fault_category", "description"],
    num_results=3,
)
for row in results.get("result", {}).get("data_array", []):
    print(f"  {row[0]} | {row[1]} | {row[2]} | {row[3][:80]}...")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Verification Summary

# COMMAND ----------

from pyspark.sql import Row

tables = ["fleet_machines", "telemetry_readings", "telemetry_daily", "maintenance_logs"]
summary = [Row(table=t, rows=spark.table(f"{CATALOG}.{SCHEMA}.{t}").count()) for t in tables]
display(spark.createDataFrame(summary))

# COMMAND ----------

# MAGIC %md
# MAGIC **Setup complete.** The conflict seed is in place:
# MAGIC - OLTP `fleet_machines`: EXC-0342 → `status = 'offline'` (real-time)
# MAGIC - OLAP `telemetry_daily`: EXC-0342 → `last_known_status = 'online'` (15-min stale)
# MAGIC
# MAGIC Proceed to `03_fleet_benchmark` to run the evaluation.
