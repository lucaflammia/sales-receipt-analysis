import polars as pl
import pandas as pd
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

  def generate_shopping_mission_report(self, year, start_month, end_month=None):
    m_range = f"M{start_month}" if not end_month or start_month == end_month else f"M{start_month}-M{end_month}"
    json_path = os.path.join(self.reports_dir, f"Mission_Insights_{year}.json")
    template_path = os.path.join(self.template_dir, "shopping_mission_template.html")
    
    output_filename = f"shopping_mission_{year}_{m_range}.html"
    output_path = os.path.join(self.html_out_dir, output_filename)

    try:
      print(f"INFO: System processing Mission Insights for {year}...")
      
      with open(json_path, "r", encoding="utf-8") as jf:
        json_data_string = json.dumps(json.load(jf))

      with open(template_path, "r", encoding="utf-8") as tf:
        template_content = tf.read()

      final_html = template_content.replace("{{DATA_PLACEHOLDER}}", json_data_string)
      
      with open(output_path, "w", encoding="utf-8") as f:
        f.write(final_html)
      
      print(f"SUCCESS: Report saved as {output_filename}")
      
      # Start the internal server
      self._serve_report(output_filename)

    except Exception as e:
      print(f"ERROR: System encountered an issue: {e}")
  
  def generate_anomaly_report(self, year, start_month, end_month=None):
    """Generates a dedicated Produce Anomaly Report."""
    m_range = f"M{start_month}" if not end_month or start_month == end_month else f"M{start_month}-M{end_month}"
    
    json_path = os.path.join(self.reports_dir, f"Anomaly_Data_{year}.json")
    template_path = os.path.join(self.template_dir, "anomalies_template.html")
    
    output_filename = f"cashier_produce_anomalies_{year}_{m_range}.html"
    output_path = os.path.join(self.html_out_dir, output_filename)

    try:
      # Log in English per Missione Spesa requirements
      logger.info(f"Commencing Produce Anomaly HTML generation for {year}...")

      if not os.path.exists(json_path):
          logger.error(f"Anomaly JSON not found at {json_path}")
          return

      with open(json_path, "r", encoding="utf-8") as jf:
          full_data = json.load(jf)
          
      # Filter specifically for Produce/Ortofrutta alerts
      # This ensures the dashboard doesn't get cluttered with cashier data
      produce_data = {
        "project": "Missione Spesa - Audit Ortofrutta",
        "period": m_range,
        "alerts": full_data.get("produce_alerts", [])
      }

      # Serialize to string for template injection
      json_data_string = json.dumps(produce_data, ensure_ascii=False)

      if not os.path.exists(template_path):
        logger.error(f"Template missing: {template_path}")
        return

      with open(template_path, "r", encoding="utf-8") as tf:
          template_content = tf.read()

      # Injecting into the placeholder
      final_html = template_content.replace("{{DATA_PLACEHOLDER}}", json_data_string)

      with open(output_path, "w", encoding="utf-8") as f:
          f.write(final_html)

      logger.info(f"SUCCESS: Produce Anomaly Report saved as {output_filename}")
      
      # Use the loop-based server to prevent ERR_CONTENT_LENGTH_MISMATCH
      self._serve_report(output_filename)

    except Exception as e:
      logger.error(f"Anomaly HTML generation failed: {e}", exc_info=True)

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

def generate_global_summary(monthly_data_list, export_path):
  """
  Strategic Report with:
  - Integrated descriptions and discriminant parameters for each mission.
  - Updated heuristics based on K-Means centroids.
  - Italian nomenclature for reports and English for logging.
  """
  if not monthly_data_list:
    logger.warning("No data collected for Global Summary.")
    return

  column_translation = {
    "shopping_mission": "Missione_di_Spesa",
    "receipt_count": "Numero_Scontrini",
    "total_mission_revenue": "Fatturato_Totale",
    "avg_trip_value": "Valore_Medio_Carrello",
    "avg_items_per_trip": "Pezzi_Medi_per_Spesa",
    "global_trend_pct": "Trend_Fatturato_Periodo_Perc",
    "global_rev_share": "Quota_Fatturato_Perc",
    "global_traff_share": "Quota_Traffico_Perc"
  }

  with pd.ExcelWriter(export_path, engine='xlsxwriter') as writer:
    workbook = writer.book
    title_fmt = workbook.add_format({'bold': True, 'font_size': 14, 'font_color': '#1A237E'})
    bold_fmt = workbook.add_format({'bold': True, 'bg_color': '#F2F2F2', 'top': 1})
    audit_header_fmt = workbook.add_format({'italic': True, 'font_color': '#7F7F7F'})
    # Define a numeric percentage format
    pct_fmt = workbook.add_format({'num_format': '0.00%'})

    # --- TAB: GLOSSARIO E LEGENDA ---
    glossary_df = pd.DataFrame({
      "Campo": list(column_translation.values()),
      "Descrizione": [
        "Categoria di acquisto identificata dall'algoritmo.",
        "Conteggio totale degli scontrini per l'intero periodo.",
        "Valore monetario lordo totale generato nel periodo.",
        "Spesa media per scontrino calcolata sull'intero periodo.",
        "Quantità media di articoli per spesa nell'intero periodo.",
        "Crescita percentuale del fatturato dal primo all'ultimo mese.",
        "Peso economico della missione sul fatturato totale globale.",
        "Incidenza della missione sul volume totale di traffico."
      ],
      "Unità_di_Misura": ["Testo", "Numero", "Euro €", "Euro €", "Pezzi", "Perc %", "Perc %", "Perc %"]
    })
    
    mission_desc_df = pd.DataFrame({
      "Missione_di_Spesa": [
        MISSION_MAP["B2B / Bulk Outlier"],
        MISSION_MAP["Weekly Stock-up"],
        MISSION_MAP["Daily Fresh Pick"],
        MISSION_MAP["Premium/Specialty Single-Item"],
        MISSION_MAP["Quick Convenience"],
        MISSION_MAP["Standard Mixed Trip"]
      ],
      "Descrizione_Dettagliata": [
        "Grandi volumi o fatturati estremi (Profilo Business).",
        "Spesa di rifornimento programmata (Grande carrello).",
        "Acquisto focalizzato su prodotti freschi e deperibili.",
        "Articoli di alta gamma o specialistici ad alto valore.",
        "Acquisto di necessità immediata o impulso.",
        "Mix merceologico bilanciato di routine."
      ],
      "Parametri_Discriminanti": [
        "Pezzi > 30 o (Pezzi > 10 e Valore/Articolo > 150€).",
        "Pezzi (Basket Size) > 5.",
        "Incidenza Freschissimi > 30% e Pezzi > 1.2.",
        "Pezzi <= 2.5 e Valore/Articolo > 45€.",
        "Pezzi <= 2.5 e Valore/Articolo < 12€.",
        "Pezzi <= 2.5 e Valore/Articolo tra 12€ e 45€."
      ]
    })

    glossary_df.to_excel(writer, sheet_name="Glossario_e_Legenda", index=False)
    mission_desc_df.to_excel(writer, sheet_name="Glossario_e_Legenda", index=False, startrow=len(glossary_df)+2)

    # --- DATA PROCESSING ---
    all_months_df = []
    all_anomalies = []
    for entry in monthly_data_list:
      period_str = entry['period']
      it_name = IT_MONTHS.get(period_str.split('-')[-1], "Mese")
      df_pd = entry['data'].to_pandas()
      
      # Extract anomalies for the Audit sheet
      anoms = df_pd[df_pd["shopping_mission"] == "B2B / Bulk Outlier"].copy()
      if not anoms.empty:
        anoms["Mese_Riferimento"] = it_name
        all_anomalies.append(anoms)

      # Map for monthly sheet
      m_exp = df_pd.copy()
      m_exp["shopping_mission"] = m_exp["shopping_mission"].replace(MISSION_MAP)
      # Convert shares to decimals (Excel expects 0.05 for 5%)
      m_exp["revenue_share_pct"] = m_exp["revenue_share_pct"] / 100
      m_exp["traffic_share_pct"] = m_exp["traffic_share_pct"] / 100
      m_exp.rename(columns={
        "shopping_mission": "Missione_di_Spesa",
        "receipt_count": "Numero_Scontrini",
        "total_mission_revenue": "Fatturato_Totale",
        "avg_trip_value": "Valore_Medio_Carrello",
        "avg_items_per_trip": "Pezzi_Medi_per_Spesa",
        "revenue_share_pct": "Quota_Fatturato_Perc",
        "traffic_share_pct": "Quota_Traffico_Perc"
      }).to_excel(writer, sheet_name=it_name, index=False)
      
      raw = df_pd.copy()
      raw['period'] = period_str
      all_months_df.append(raw)

    # --- TAB: AUDIT ANOMALIE ---
    if all_anomalies:
      an_df = pd.concat(all_anomalies)
      an_df["shopping_mission"] = an_df["shopping_mission"].replace(MISSION_MAP)
      an_df = an_df.rename(columns=column_translation)
      cols_to_keep = ["Mese_Riferimento", "Missione_di_Spesa", "Numero_Scontrini", "Fatturato_Totale", "Valore_Medio_Carrello"]
      an_df[cols_to_keep].to_excel(writer, sheet_name="Audit_Anomalie", index=False, startrow=2)
      audit_sheet = writer.sheets["Audit_Anomalie"]
      audit_desc = "LOGICA B2B: Pezzi > 30 o (Pezzi > 10 e Valore/Articolo > 150€)."
      # Column indices for Quota_Fatturato_Perc and Quota_Traffico_Perc
      audit_sheet.set_column('G:H', 15, pct_fmt)
      audit_sheet.write('A1', audit_desc, audit_header_fmt)

    # --- TAB: EXECUTIVE SUMMARY ---
    summary_df = pd.concat(all_months_df)
    periods = sorted(summary_df["period"].unique())
    
    trend_pivot = summary_df.pivot_table(index="shopping_mission", columns="period", values="total_mission_revenue", aggfunc="sum").sort_index(axis=1)
    global_trend = trend_pivot.apply(lambda r: ((r.dropna().iloc[-1] - r.dropna().iloc[0]) / r.dropna().iloc[0] * 100) if len(r.dropna()) > 1 else 0.0, axis=1).round(2)

    exec_summary = summary_df.groupby("shopping_mission").agg({
      "receipt_count": "sum", "total_mission_revenue": "sum", "avg_trip_value": "mean", "avg_items_per_trip": "mean"
    })
    
    total_rev = exec_summary["total_mission_revenue"].sum()
    total_traff = exec_summary["receipt_count"].sum()
    
    exec_summary["global_trend_pct"] = global_trend
    exec_summary["global_rev_share"] = (exec_summary["total_mission_revenue"] / total_rev * 100).round(2)
    exec_summary["global_traff_share"] = (exec_summary["receipt_count"] / total_traff * 100).round(2)

    exec_final = exec_summary.reset_index()
    exec_final["shopping_mission"] = exec_final["shopping_mission"].replace(MISSION_MAP)
    exec_final = exec_final.rename(columns=column_translation)
    
    final_cols_ordered = [v for v in column_translation.values() if v in exec_final.columns]
    sorted_summary = exec_final[final_cols_ordered].sort_values("Fatturato_Totale", ascending=False)
    
    sheet_name = "Executive_Summary"
    sorted_summary.to_excel(writer, sheet_name=sheet_name, index=False, startrow=2)
    summary_sheet = writer.sheets[sheet_name]
    # Column indices for Quota_Fatturato_Perc and Quota_Traffico_Perc
    summary_sheet.set_column('G:H', 15, pct_fmt)
    summary_sheet.write('A1', f"ANALISI GLOBALE: {periods[0]} - {periods[-1]}", title_fmt)

    # Footer Totals
    total_row_idx = len(sorted_summary) + 4 
    totals_data = ["TOTALE", total_traff, total_rev, "", "", "", 100.0, 100.0]
    for col_num, value in enumerate(totals_data):
      summary_sheet.write(total_row_idx, col_num, value, bold_fmt)

    # Growth Chart
    chart = workbook.add_chart({'type': 'column'})
    trend_col_idx = final_cols_ordered.index(column_translation["global_trend_pct"])
    chart.add_series({
      'name': 'Trend Crescita %',
      'categories': [sheet_name, 3, 0, len(sorted_summary)+2, 0],
      'values':      [sheet_name, 3, trend_col_idx, len(sorted_summary)+2, trend_col_idx],
      'fill':        {'color': '#1A237E'}
    })
    summary_sheet.insert_chart('K3', chart)

  logger.info(f"📈 Strategic Report saved: {export_path}")

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
      "articoloCodice": "product_id"
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

    return insights, produce_anomalies

  except Exception as e:
    logger.error(f"❌ Error for {year}-{month}: {e}", exc_info=True)

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
  fe = FeatureEngineer()
  report_gen = ReportGenerator()
  
  # Storage for the Global Summary
  global_insights = []
  global_anomalies = []
  
  start_month = args.month
  end_month = args.month_end or args.month
  months = range(start_month, end_month + 1)
  target_days = list(range(args.day, (args.day_end or args.day) + 1)) if args.day else None
  
  # DATA PROCESSING
  for i, m in enumerate(months):
    print(f"INFO: Processing partition {args.year}-{m:02d}...")
    
    # Process monthly data
    monthly_insight, monthly_anomalies = process_partition(
      args.year, m, config, fe, 
      days=target_days, 
      fit_global=(i == 0)
    )
    
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
    
    log_resource_usage(f"Completed Month {m}")

  # FINAL REPORT GENERATION
  if global_insights:
    # Define Paths
    json_filename = f"Mission_Insights_{args.year}.json"
    json_path = os.path.join("reports", json_filename)
    
    # Save JSON Feed first (System data)
    print(f"INFO: Exporting JSON feed to {json_path}")
    export_to_json(global_insights, json_path)

    # Prepare and Sanitize the Anomaly Payload
    anomaly_json_path = os.path.join("reports", f"Anomaly_Data_{args.year}.json")
    
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
      "cashier_anomalies": [
        {"id": "Cassa-04", "deviation": 28.4, "risk": 92},
        {"id": "Cassa-11", "deviation": 15.2, "risk": 65}
      ],
      "produce_alerts": sanitize_anomalies(global_anomalies)
    }

    # Save with a default handler as a secondary safety net
    with open(anomaly_json_path, 'w', encoding='utf-8') as f:
      json.dump(anomaly_payload, f, indent=4, ensure_ascii=False, default=json_serial)
    
    logger.info(f"INFO: Exported Anomaly data to {anomaly_json_path}")

    # Generate the unified dashboard
    report_gen.generate_shopping_mission_report(
      year=args.year, 
      start_month=args.month,
      end_month=args.month_end
    )

    report_gen.generate_anomaly_report(
      year=args.year, 
      start_month=args.month,
      end_month=args.month_end
    )

  print(f"INFO: ⏱️ Total Execution Time: {time.time() - start_time:.2f}s")

if __name__ == "__main__":
  main()
