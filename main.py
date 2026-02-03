import polars as pl
import logging
import psutil
import time
import os
import argparse
import datetime
import webbrowser
import http.server
import socketserver
import glob
import json
from src.ingestion import load_config
from src.engineering import FeatureEngineer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

IT_MONTHS = {
  "01": "Gennaio", "02": "Febbraio", "03": "Marzo", "04": "Aprile",
  "05": "Maggio", "06": "Giugno", "07": "Luglio", "08": "Agosto",
  "09": "Settembre", "10": "Ottobre", "11": "Novembre", "12": "Dicembre"
}

MISSION_MAP = {
  "Standard Mixed Trip": "Spesa Standard Mista",
  "Premium/Specialty Single-Item": "Premium / Specialità Singolo Articolo",
  "B2B / Bulk Outlier": "B2B / Ingrosso Outlier",
  "Weekly Stock-up": "Spesa Settimanale di Scorta",
  "Quick Convenience": "Convenienza Rapida",
  "Daily Fresh Pick": "Fresco Quotidiano"
}

class ReportGenerator:
  def __init__(self, base_dir="/workspaces/sales-receipt-analysis"):
    self.base_dir = base_dir
    self.html_out_dir = os.path.join(base_dir, "html_reports")
    self.template_dir = os.path.join(base_dir, "templates")
    self.reports_dir = os.path.join(base_dir, "reports")

    for directory in [self.html_out_dir, self.reports_dir]:
      os.makedirs(directory, exist_ok=True)

  def _serve_report(self, filename, start_port=8000):
    """Starts a local server and keeps it alive to serve all dashboard assets."""
    os.chdir(self.html_out_dir)
    handler = http.server.SimpleHTTPRequestHandler
    
    port = start_port
    while port < 8010:
      try:
          socketserver.TCPServer.allow_reuse_address = True
          with socketserver.TCPServer(("", port), handler) as httpd:
              url = f"http://localhost:{port}/{filename}"
              print("\n" + "--- REPORT READY (Missione Spesa) ---".center(50))
              print(f"URL: {url}")
              print("ACTION: Ctrl+Click to open. Press Ctrl+C in terminal to stop server.")
              print("-" * 50 + "\n")
              
              webbrowser.open(url)
              
              # Instead of handle_request(), use serve_forever() 
              # so the browser can fetch the HTML, CSS, and JSON data fully.
              try:
                  httpd.serve_forever()
              except KeyboardInterrupt:
                  print("\nStopping server...")
                  httpd.shutdown()
          break
      except OSError:
          port += 1

  def generate_unified_dashboard(self, year, start_month, end_month=None):
    """
    Generates a single, standalone HTML dashboard with two tabs:
    1. Missione Spesa (Shopping Missions)
    2. Anomalie Ortofrutta (Produce Anomalies)
    """
    m_range = f"M{start_month}" if not end_month or start_month == end_month else f"M{start_month}-M{end_month}"
    
    # Paths for localized data sources
    mission_json_path = os.path.join(self.reports_dir, f"Mission_Insights_{year}.json")
    anomaly_json_path = os.path.join(self.reports_dir, f"Anomaly_Data_{year}.json") 
    master_tpl_path = os.path.join(self.template_dir, "dashboard_template.html")
    
    output_filename = f"unified_dashboard_{year}_{m_range}.html"
    output_path = os.path.join(self.html_out_dir, output_filename)

    try:
      # Initialize Combined Data Structure
      combined_data = {
        "metadata": {
          "anno": year, 
          "periodo": m_range, 
          "data_generazione": datetime.datetime.now().strftime("%d/%m/%Y %H:%M"),
          "titolo_dashboard": "Dashboard Missione Spesa & Anomalie"
        },
        "tabs": {
          "missioni": {
            "label": "Risultati Missione Spesa",
            "data": {}
          },
          "anomalie": {
            "label": "Anomalie Prodotti Ortofrutta",
            "data": {}
          }
        }
      }
      
      # Load and Map Mission Data
      if os.path.exists(mission_json_path):
        with open(mission_json_path, "r", encoding="utf-8") as f:
          # We store this under the Italian key for the dashboard tab
          combined_data["tabs"]["missioni"]["data"] = json.load(f)
      
      # Load and Map Anomaly Data
      if os.path.exists(anomaly_json_path):
        with open(anomaly_json_path, "r", encoding="utf-8") as f:
          combined_data["tabs"]["anomalie"]["data"] = json.load(f)

      # Inject into HTML Template
      if not os.path.exists(master_tpl_path):
        raise FileNotFoundError(f"Master template not found at {master_tpl_path}")

      with open(master_tpl_path, "r", encoding="utf-8") as f:
        template_content = f.read()

      # Serialization for the JS frontend
      json_payload = json.dumps(combined_data, ensure_ascii=False)
      final_html = template_content.replace("{{DASHBOARD_DATA_JSON}}", json_payload)
      
      # Save Final Dashboard
      with open(output_path, "w", encoding="utf-8") as f:
        f.write(final_html)

      # English log as per instructions
      logger.info(f"SUCCESS: Unified two-tab dashboard generated: {output_filename}")
      
      # Serve/Open the report
      self._serve_report(output_filename)

    except Exception as e:
      logger.error(f"ERROR: Failed to generate unified dashboard: {e}", exc_info=True)

def log_resource_usage(stage: str):
  process = psutil.Process(os.getpid())
  mem = process.memory_info().rss / (1024 * 1024)
  logger.info(f"[{stage}] RAM: {mem:.2f} MB")

def find_partition_path(area_root, year, month):
  pattern = os.path.join(area_root, "**", f"year={year}", f"month={month}")
  matches = glob.glob(pattern, recursive=True)
  dirs = [m for m in matches if os.path.isdir(m)]
  return dirs[0] if dirs else None

def export_to_json(monthly_data_list, export_path):
  """
  Generates a JSON feed for LLM consumption.
  """
  json_output = {
    "metadata": {
      "project": "Missione Spesa",
      "analysis_date": time.strftime("%Y-%m-%d"),
    },
    "monthly_insights": []
  }

  for entry in monthly_data_list:
    df_pd = entry['data'].to_pandas()
    df_pd["shopping_mission"] = df_pd["shopping_mission"].replace(MISSION_MAP)
    
    json_output["monthly_insights"].append({
      "period": entry['period'],
      "results": df_pd.to_dict(orient="records")
    })

  with open(export_path, 'w', encoding='utf-8') as f:
    json.dump(json_output, f, indent=4, ensure_ascii=False)
  logger.info(f"📁 LLM-Ready JSON saved: {export_path}")

def process_partition(year, month, config, feature_engineer, days=None, fit_global=False):
  env_config = config[config["environment"]]
  area_code = env_config.get('area_code', '382')
  
  area_path = os.path.join(
    env_config["data_path"], 
    "raw_normalized", "venduto", f"area={area_code}"
  )

  base_data_path = find_partition_path(area_path, year, month)
  
  if not base_data_path:
    logger.warning(f"⚠️ Path not found for Year={year}, Month={month}")
    return

  logger.info(f"📂 Found Data: {base_data_path}")

  all_files = []
  if days:
    for d in days:
      day_glob = os.path.join(base_data_path, f"day={d}", "*.parquet")
      all_files.extend(glob.glob(day_glob))
  else:
    full_glob = os.path.join(base_data_path, "day=*", "*.parquet")
    all_files = glob.glob(full_glob)

  if not all_files:
    logger.warning(f"⚠️ No parquet files found in {base_data_path}")
    return

  logger.info(f"📊 Found {len(all_files)} files to scan.")

  try:
    # Load LazyFrame
    raw_data_lazy = pl.scan_parquet(all_files)
    
    # Schema Normalization
    raw_data_lazy = raw_data_lazy.with_columns([
      pl.col("scontrinoIdentificativo").cast(pl.String),
      pl.col("articoloCodice").cast(pl.String)
    ]).rename({
      "scontrinoIdentificativo": "receipt_id",
      "totaleLordo": "line_total",
      "quantitaPeso": "weight",
      "negozioCodice": "store_id",
      "scontrinoData": "date_str",
      "scontrinoOra": "time_str",
      "articoloCodice": "product_id",
      "cassiereCodice": "cashier_id"
    })

    # Feature Engineering
    raw_data_lazy = raw_data_lazy.with_columns([
      pl.lit(f"{year}-{month:02d}").alias("partition_key"),
      pl.col("line_total").median().over("product_id").alias("Standard_Price")
    ]).with_columns([
      (pl.col("line_total") < pl.col("Standard_Price")).fill_null(False).alias("is_discounted"),
      (pl.col("line_total") - pl.col("Standard_Price")).fill_null(0.0).alias("price_delta"),
      pl.when(pl.col("weight") > 0).then(pl.col("line_total") / pl.col("weight"))
        .otherwise(pl.col("line_total")).fill_null(0.0).alias("avg_unit_price")
    ])

    # DIAGNOSTIC: Check if we actually have multi-item baskets before aggregating
    test_counts = raw_data_lazy.group_by("receipt_id").len().collect()
    avg_lines = test_counts["len"].mean()
    logger.info(f"🔍 Diagnostic: Average lines per receipt: {avg_lines:.2f}")

    # Feature Aggregation
    df_lazy = feature_engineer.extract_canonical_features(raw_data_lazy)
    df = df_lazy.collect()

    # Clean data for ALL subsequent ML steps
    mission_features = ["basket_size", "basket_value_per_item", "freshness_weight_ratio"]
    mission_features = [f for f in mission_features if f in df.columns]
    
    # Critical: Fill nulls here so select_features_rf doesn't see an empty array
    df = df.with_columns([
      pl.col(mission_features).fill_null(0.0).fill_nan(0.0),
      pl.col("basket_value").fill_null(0.0)
    ])

    # --- ELBOW METHOD DIAGNOSTIC ---
    if fit_global:
      logger.info("📊 Calculating Elbow Point to validate cluster count...")
      # This uses the method in engineering.py to find where the WCSS curve bends
      optimal_k = feature_engineer.determine_elbow_method(df, features=mission_features)
      
      if optimal_k != 6:
        logger.warning(f"📉 Elbow Analysis suggests {optimal_k} clusters, but we are proceeding with 6 for project consistency.")
      else:
        logger.info("✅ Elbow Analysis confirms 6 clusters is optimal for this data.")
    # -------------------------------

    logger.info(f"🧠 Optimizing Shopping Missions for {df.height} receipts...")

    # Prepare features for Unsupervised Learning
    mission_features = ["basket_size", "basket_value_per_item", "freshness_weight_ratio"]
    mission_features = [f for f in mission_features if f in df.columns]
    df = df.with_columns([pl.col(mission_features).fill_null(0.0).fill_nan(0.0)])
    
    logger.info("Starting Mission Segmentation...")
    # Perform Clustering with Optimized Centroids
    df = feature_engineer.segment_shopping_missions(
      df, 
      features=mission_features, 
      fit_global=fit_global
    )

    # --- ANOMALY SEVERITY AUDIT ---
    # Flagging B2B/Outliers for separate technical audit
    anomalies = df.filter(pl.col("shopping_mission") == "B2B / Bulk Outlier")

    if not anomalies.is_empty():
      audit_path = f"audit/b2b_audit_{year}_{month}.csv"
      os.makedirs("audit", exist_ok=True)
      
      # Calculate severity: How many standard deviations away is the revenue?
      avg_rev = df["basket_value"].mean()
      std_rev = df["basket_value"].std()
      
      anomalies = anomalies.with_columns([
        ((pl.col("basket_value") - avg_rev) / std_rev).alias("anomaly_score"),
        pl.lit(f"{year}-{month}").alias("period")
      ])
      
      anomalies.write_csv(audit_path)
      logger.warning(f"🚩 {len(anomalies)} B2B anomalies detected! Audit report saved to: {audit_path}")
    # ----------------------------------

    # ANOMALY DETECTION (Cassa & Ortofrutta)
    # Sweethearting & Cashier Performance
    if "cashier_id" in df.columns:
      logger.info("Running Cashier Integrity Audit (Sweethearting)...")
      df = feature_engineer.detect_cashier_anomalies(df)

    # Produce Weighing Errors (Ortofrutta)
    # We pass the raw line-item data to check price/weight ratios
    logger.info("Running Produce (Ortofrutta) Weight Audit...")
    produce_anomalies = feature_engineer.detect_produce_weighing_errors(raw_data_lazy.collect())

    logger.info("Running Cashier Integrity Audit (Sweethearting)...")

    # Ensure cashier_id exists
    if "cashier_id" not in df.columns:
      logger.warning("Log: cashier_id not found in aggregated features, forcing N/A")
      df = df.with_columns(pl.lit("N/A").alias("cashier_id"))
    
    df = feature_engineer.detect_cashier_anomalies(df)
    
    # Filter for the report: Score -1 is the outlier flag, Z > 1.5 is the visibility threshold
    cashier_alerts = df.select([
      "cashier_id", "cashier_anomaly_score", "cashier_z_score"
    ]).unique().filter(
      (pl.col("cashier_anomaly_score") == -1) | (pl.col("cashier_z_score") > 1.5)
    )
    
    if cashier_alerts.is_empty():
      logger.info("Log: Still no significant cashier anomalies even with relaxed thresholds.")
    else:
      logger.info(f"Log: ALERT! {cashier_alerts.height} cashiers flagged for review.")

    # Anomaly Detection (Post-Clustering)
    # Using the same features to see which specific missions contain outliers
    rf_imp = feature_engineer.select_features_rf(df, features=mission_features, target="basket_value")
    lasso_imp = feature_engineer.select_features_lasso(df, features=mission_features, target="basket_value")
    consensus = feature_engineer.get_consensus_features(rf_imp, lasso_imp)

    if consensus:
      # Run only during the first month (fit_global) to validate the feature set
      if fit_global:
        logger.info("Stat Check: Validating feature independence (VIF)...")
        # We use the keys from consensus to check only the selected features
        vif_results = feature_engineer.calculate_vif(df, features=list(consensus.keys()))
        
        for feat, score in vif_results.items():
          status = "PASS" if score < 5 else "HIGH COLLINEARITY"
          logger.info(f"[VIF Analysis] {feat}: {score:.2f} - {status}")
      df = feature_engineer.add_anomaly_score(df, features=list(consensus.keys()))
      df = feature_engineer.map_severity(df)

    # Generate Strategic Dashboard Insights
    grand_total_revenue = df["basket_value"].sum()
    total_receipts = df.height

    insights = (
      df.group_by("shopping_mission")
      .agg([
        pl.len().alias("receipt_count"),
        pl.col("basket_value").sum().alias("total_mission_revenue"),
        pl.col("basket_value").mean().round(2).alias("avg_trip_value"),
        pl.col("basket_size").mean().round(2).alias("avg_items_per_trip")
      ])
      .with_columns([
        ((pl.col("total_mission_revenue") / grand_total_revenue) * 100).round(3).alias("revenue_share_pct"),
        ((pl.col("receipt_count") / total_receipts) * 100).round(3).alias("traffic_share_pct")
      ])
      .sort("revenue_share_pct", descending=True)
    )

    plot_insights = insights.clone().to_pandas()
    plot_insights["shopping_mission"] = plot_insights["shopping_mission"].replace(MISSION_MAP)

    # Visualization - Ensure the path and labels are ready
    os.makedirs("plots", exist_ok=True)
    feature_engineer.plot_mission_impact(
      plot_insights,
      path=os.path.join("plots", f"impact_{year}_{month}.png"),
      title=f"Analisi Missioni: {IT_MONTHS.get(f'{month:02d}', 'Mese')} {year}"
    )

    # Export
    export_root = os.path.join(env_config["processed_data_path"], f"area={area_code}", f"year={year}", f"month={month}")
    if days:
      export_root = os.path.join(export_root, f"days_{min(days)}_to_{max(days)}")

    os.makedirs(export_root, exist_ok=True)
    df.write_parquet(os.path.join(export_root, "canonical_baseline.parquet"))
    
    logger.info("\n" + "💰 STRATEGIC MISSION REPORT".center(60, "="))
    print(insights)
    logger.info("="*60)

    return insights, produce_anomalies, cashier_alerts

  except Exception as e:
    logger.error(f"❌ Error for {year}-{month}: {e}", exc_info=True)
    return None, None, None

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--year", type=int, required=True)
  parser.add_argument("--month", type=int, required=True)
  parser.add_argument("--month_end", type=int)
  parser.add_argument("--day", type=int)
  parser.add_argument("--day_end", type=int)
  args = parser.parse_args()

  start_time = time.time()
  config = load_config()
  report_gen = ReportGenerator()
  
  # Storage for the Global Summary
  global_insights = []
  global_anomalies = []
  global_cashier_alerts = []
  
  start_month = args.month
  end_month = args.month_end or args.month
  months = range(start_month, end_month + 1)
  target_days = list(range(args.day, (args.day_end or args.day) + 1)) if args.day else None
  
  # DATA PROCESSING
  for i, m in enumerate(months):
    print(f"INFO: Processing partition {args.year}-{m:02d}...")

   # Fresh state for every month to prevent cumulative drift
    current_fe = FeatureEngineer()
    
    # Process monthly data
    monthly_insight, monthly_anomalies, monthly_cashier_alerts = process_partition(
      args.year, m, config, current_fe, 
      days=target_days, 
      fit_global=True # Usually safer to fit per-month for produce weighing
    )

    # Check if the partition actually returned data
    if monthly_insight is None:
      logger.error(f"Partition {args.year}-{m:02d} failed. Skipping...")
      continue
    
    if monthly_insight is not None:
      global_insights.append({
        "period": f"{args.year}-{m:02d}", 
        "data": monthly_insight
      })

    if monthly_anomalies is not None:
      # Convert Polars to Dict for JSON export
      global_anomalies.append({
        "period": f"{args.year}-{m:02d}",
        "details": monthly_anomalies.to_dicts()
      })

    if monthly_cashier_alerts is not None:
      global_cashier_alerts.append({
        "period": f"{args.year}-{m:02d}",
        "details": monthly_cashier_alerts.to_dicts()
      })
    
    log_resource_usage(f"Completed Month {m}")

  # FINAL REPORT GENERATION
  if global_insights:
    # Define Paths
    json_mission_path = os.path.join("reports", f"Mission_Insights_{args.year}.json")
    json_anomaly_path = os.path.join("reports", f"Anomaly_Data_{args.year}.json")
    
    # Save JSON Feed first (System data)
    print(f"INFO: Exporting JSON feed to {json_mission_path}")
    export_to_json(global_insights, json_mission_path)

    def json_serial(obj):
      """JSON serializer for objects not serializable by default json code"""
      if isinstance(obj, (datetime.date, datetime.datetime)):
        return obj.isoformat()
      raise TypeError(f"Type {type(obj)} not serializable")

    def sanitize_anomalies(anomaly_list):
      sanitized = []
      for entry in anomaly_list:
        # Ensure we handle the details list safely
        details = entry.get("details", [])
        
        # If it's a list of dicts (from Polars to_dicts())
        if isinstance(details, list):
          for row in details:
            for key, value in row.items():
              # Convert dates/datetimes to ISO strings
              if isinstance(value, (datetime.date, datetime.datetime)):
                row[key] = value.isoformat()
        
        sanitized.append({
          "period": entry["period"],
          "details": details
        })
      return sanitized

    # Aggregate the payload
    anomaly_payload = {
      "cashier_anomalies": sanitize_anomalies(global_cashier_alerts),
      "produce_alerts": sanitize_anomalies(global_anomalies)
    }

    # Save with a default handler as a secondary safety net
    with open(json_anomaly_path, 'w', encoding='utf-8') as f:
      json.dump(anomaly_payload, f, indent=4, ensure_ascii=False, default=json_serial)
    
    logger.info(f"INFO: Exported Anomaly data to {json_anomaly_path}")

    # Generate the unified dashboard
    report_gen.generate_unified_dashboard(
      year=args.year, 
      start_month=args.month,
      end_month=args.month_end
    )

  print(f"INFO: ⏱️ Total Execution Time: {time.time() - start_time:.2f}s")

if __name__ == "__main__":
  main()
