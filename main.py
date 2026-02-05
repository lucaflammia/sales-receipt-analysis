import polars as pl
import logging
import psutil
import time
import os
import argparse
import datetime
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

def log_resource_usage(stage: str):
  process = psutil.Process(os.getpid())
  mem = process.memory_info().rss / (1024 * 1024)
  logger.info(f"[{stage}] RAM: {mem:.2f} MB")

def find_partition_path(area_root, year, month):
  pattern = os.path.join(area_root, "**", f"year={year}", f"month={month}")
  matches = glob.glob(pattern, recursive=True)
  dirs = [m for m in matches if os.path.isdir(m)]
  return dirs[0] if dirs else None

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
    return None, None, None, None, None

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
    
    # Processing the lazy frame with schema-aligned mappings
    raw_data_lazy = raw_data_lazy.with_columns([
      # Cast identifiers to String for join stability
      pl.col("scontrinoIdentificativo").cast(pl.String),
      pl.col("articoloCodice").cast(pl.String),

      # EXTRACT HOUR OF DAY: Parsing 'scontrinoOra' (HH:mm) 
      # This allows for mean() and std() calculations in the fingerprint
      pl.col("scontrinoOra").str.strptime(pl.Time, format="%H:%M")
      .dt.hour()
      .alias("hour_of_day"),

      # Handles 'Y', 'y', '1', or 1 (as int)
      pl.col("abortito").cast(pl.String).str.to_uppercase().is_in(["Y", "1", "S"]).alias("is_abort"),
      pl.col("storno").cast(pl.String).str.to_uppercase().is_in(["Y", "1", "S"]).alias("is_item_void"),
      pl.col("annullo").cast(pl.String).str.to_uppercase().is_in(["Y", "1", "S"]).alias("is_void"),
      pl.col("omaggio").cast(pl.String).str.to_uppercase().is_in(["Y", "1", "S"]).alias("is_gift")

    ]).rename({
      "scontrinoIdentificativo": "receipt_id",
      "totaleLordo": "line_total",
      "quantitaPeso": "weight",
      "negozioCodice": "store_id",
      "scontrinoData": "date_str",
      "scontrinoOra": "time_str",
      "articoloCodice": "product_id",
      "cassiereCodice": "cashier_id",
      "formaPagamentoCodice": "payment_method_code",
      "totaleSconti": "total_discounts",
      "puntiRitiro": "points_redeemed"
    })

     # --- SHOPPING MISSION ANALYSIS: TRADITIONAL MISSION SEGMENTATION ------

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

    # Check validation of metrics
    feature_engineer.validate_business_metrics(df)

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

    # Flagging B2B/Outliers for separate technical audit
    b2b_anomalies = df.filter(pl.col("shopping_mission") == "B2B / Bulk Outlier")

    if not b2b_anomalies.is_empty():
      audit_path = f"audit/b2b_audit_{year}_{month}.csv"
      os.makedirs("audit", exist_ok=True)
      
      # Calculate severity: How many standard deviations away is the revenue?
      avg_rev = df["basket_value"].mean()
      std_rev = df["basket_value"].std()
      
      b2b_anomalies = b2b_anomalies.with_columns([
        ((pl.col("basket_value") - avg_rev) / std_rev).alias("anomaly_score"),
        pl.lit(f"{year}-{month}").alias("period")
      ])
      
      b2b_anomalies.write_csv(audit_path)
      logger.warning(f"🚩 {len(b2b_anomalies)} B2B anomalies detected! Audit report saved to: {audit_path}")

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

    # --- Generate Strategic Dashboard Insights ---
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

    logger.info("\n" + "💰 STRATEGIC MISSION REPORT".center(60, "="))
    print(insights)
    logger.info("="*60)
    
    # ANOMALY DETECTION (Cassa & Ortofrutta)

    # Produce Weighing Errors (Ortofrutta)
    # Pass raw line-item data to check price/weight ratios using MAD
    logger.info("Running Produce (Ortofrutta) Weight Audit...")
    produce_anomalies = feature_engineer.detect_produce_weighing_errors(raw_data_lazy.collect())

    # Cashier Integrity Audit (Cashier Sweethearting Alerts)
    logger.info("Running Cashier Integrity Audit (Cashier Sweethearting Alerts)...")

    # Ensure cashier_id exists before processing
    if "cashier_id" not in df.columns:
      logger.warning("Log: cashier_id not found in aggregated features, forcing N/A")
      df = df.with_columns(pl.lit("N/A").alias("cashier_id"))

    # Run the anomaly detection (Z-Score & Isolation Forest)
    df = feature_engineer.detect_cashier_sweethearting_anomalies(df)

    # Filter for Report (Registro Audit)
    # Score -1 (outlier flag) or Z > 1.5 (visibility threshold)
    cashier_sweethearting_anomalies = df.select([
      "cashier_id", 
      "cashier_sweethearting_anomaly_score", 
      "cashier_sweethearting_z_score"
    ]).unique().filter(
      (pl.col("cashier_sweethearting_anomaly_score") == -1) | (pl.col("cashier_sweethearting_z_score") > 1.5)
    )

    # Logging results
    if cashier_sweethearting_anomalies.is_empty():
      logger.info("Log: No significant cashier sweethearting anomalies detected.")
    else:
      logger.info(f"Log: ALERT! {cashier_sweethearting_anomalies.height} cashiers flagged for review.")

    # --- Cashier Behavioral Fingerprinting & Margin Leakage Analysis ---
    logger.info("AI STRATEGY: Starting Cashier Behavioral Fingerprinting...")
    # We collect here as fingerprinting requires aggregation across the whole partition
    df_raw = raw_data_lazy.collect()
    
    fingerprint_df = feature_engineer.extract_cashier_fingerprint(df_raw)
    monthly_cashier_fp_anomalies = feature_engineer.detect_behavioral_anomalies(fingerprint_df)

    # Filter high-risk alerts for the report (Italian names used in output table)
    monthly_cashier_fp_anomalies = monthly_cashier_fp_anomalies.filter(pl.col("risk_score") > 0.5).select([
      pl.col("cashier_id").alias("Codice Cassiere"),
      pl.col("risk_score").round(2).alias("Score Rischio Comportamentale"),
      pl.col("cash_reliance_ratio").round(3).alias("Indice Uso Contanti"),
      pl.col("void_rate_per_receipt").round(3).alias("Tasso Storni")
    ])

    logger.info(f"Cashier Behavioral Fingerprinting complete. {monthly_cashier_fp_anomalies.height} high-risk cashiers detected.")

    margin_leakage_data = feature_engineer.run_margin_leakage_radar(df_raw)

    logger.info(f"Margin Leakage Analysis complete. {margin_leakage_data.height} cashiers analyzed.")

    return insights, produce_anomalies, cashier_sweethearting_anomalies, monthly_cashier_fp_anomalies, margin_leakage_data

  except Exception as e:
    logger.error(f"❌ Error for {year}-{month}: {e}", exc_info=True)
    return None, None, None, None, None

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
  
  # Storage for the Global Summary
  global_insights = []
  global_produce_anomalies = []
  global_cashier_sweethearting_anomalies = []
  global_cashier_fp_anomalies = []
  global_margin_leakage = []
  
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
    monthly_insight, monthly_produce_anomalies, monthly_cashier_sweethearting_anomalies, monthly_cashier_fp_anomalies, monthly_margin_leakage = process_partition(
      args.year, m, config, current_fe, 
      days=target_days, 
      fit_global=False # If True is to fit per-month models
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

    if monthly_produce_anomalies is not None:
      # Convert Polars to Dict for JSON export
      global_produce_anomalies.append({
        "period": f"{args.year}-{m:02d}",
        "details": monthly_produce_anomalies.to_dicts()
      })

    if monthly_cashier_sweethearting_anomalies is not None:
      global_cashier_sweethearting_anomalies.append({
        "period": f"{args.year}-{m:02d}",
        "details": monthly_cashier_sweethearting_anomalies.to_dicts()
      })
    
      if monthly_cashier_fp_anomalies is not None:
        global_cashier_fp_anomalies.append({
          "period": f"{args.year}-{m:02d}",
          "details": monthly_cashier_fp_anomalies.to_dicts()
        })

      if monthly_margin_leakage is not None:
        global_margin_leakage.append({
          "period": f"{args.year}-{m:02d}",
          "details": monthly_margin_leakage.to_dicts()
        })
    
    log_resource_usage(f"Completed Month {m}")

  # FINAL REPORT GENERATION
  if global_insights:
    # Define Paths
    m_range = f"M{start_month}" if not end_month or start_month == end_month else f"M{start_month}-M{end_month}"
    json_mission_path = os.path.join("reports", f"Strategic_Mission_Intelligence_{args.year}_{m_range}.json")
    json_integrity_path = os.path.join("reports", f"Store_Integrity_Audit_Report_{args.year}_{m_range}.json")
    json_behavioral_path = os.path.join("reports", f"Cashier_Behavioral_Risk_DNA_{args.year}_{m_range}.json")
    
    def json_serial(obj):
      """JSON serializer for objects not serializable by default json code"""
      if isinstance(obj, (datetime.date, datetime.datetime)):
        return obj.isoformat()
      raise TypeError(f"Type {type(obj)} not serializable")

    def map_and_sanitize(anomaly_list, key_map):
      """Rename keys in italian and serialize time periods"""
      sanitized = []
      for entry in anomaly_list:
        details = entry.get("details", [])
        mapped_details = []
        if isinstance(details, list):
          for row in details:
            new_row = {}
            for k, v in row.items():
              # Rinomina se presente nel mapping, altrimenti mantiene originale
              new_key = key_map.get(k, k)
              val = v.isoformat() if isinstance(v, (datetime.date, datetime.datetime)) else v
              new_row[new_key] = val
            mapped_details.append(new_row)
        sanitized.append({"periodo": entry["period"], "dettagli": mapped_details})
      return sanitized
    
    mission_payload = []
    for entry in global_insights:
      df_pd = entry['data'].to_pandas()
      df_pd["shopping_mission"] = df_pd["shopping_mission"].replace(MISSION_MAP)
      # Ridenominazione colonne per il JSON
      df_pd = df_pd.rename(columns={
        "shopping_mission": "Missione Spesa",
        "receipt_count": "Numero Scontrini",
        "total_mission_revenue": "Fatturato Totale Missione",
        "avg_trip_value": "Valore Medio Spesa",
        "avg_items_per_trip": "Media Articoli per Spesa",
        "revenue_share_pct": "Percentuale Fatturato",
        "traffic_share_pct": "Percentuale Traffico"
      })
      mission_payload.append({"periodo": entry["period"], "risultati": df_pd.to_dict(orient="records")})

    mission_final = {
      "metadata": {
        "titolo": "Intelligence Strategica Missioni di Spesa",
        "descrizione": "Classificazione dei viaggi di spesa basata su algoritmi di segmentazione del paniere.",
        "arricchimento_business": "Questo report permette di ottimizzare il layout del punto vendita e le promozioni basandosi sul comportamento reale d'acquisto. Identificare la prevalenza di 'Missioni di Scorta' rispetto a 'Pasto Pronto' guida le decisioni sullo spazio espositivo e l'assortimento dei freschi, massimizzando il valore del carrello medio.",
        "dizionario_dati": {
          "periodo": "L'intervallo temporale (anno-mese) dell'analisi.",
          "Missione Spesa": "Etichetta del cluster (es. 'Spesa di Scorta', 'Pasto Pronto', 'Integrazione Freschi') che identifica lo scopo del cliente.",
          "Numero Scontrini": "Volume totale di atti d'acquisto riconducibili a quella specifica missione.",
          "Fatturato Totale Missione": "Somma totale incassata per tutti gli scontrini appartenenti a quel cluster.",
          "Valore Medio Spesa": "Importo medio (Scontrino Medio) per questa missione.",
          "Media Articoli per Spesa": "Numero medio di referenze (pezzi) acquistate per ogni atto d'acquisto.",
          "Percentuale Fatturato": "Quanto contribuisce questa missione al fatturato totale del negozio in termini percentuali.",
          "Percentuale Traffico": "Quanto contribuisce questa missione al numero totale di clienti (scontrini) che entrano in negozio."
        }
      },
      "insight_mensili": mission_payload
    }
    with open(json_mission_path, 'w', encoding='utf-8') as f:
      json.dump(mission_final, f, indent=4, ensure_ascii=False, default=json_serial)

    # Export Integrity Audit (Sweethearting & Produce)
    integrity_map = {
      "cashier_id": "Codice Cassiere",
      "cashier_sweethearting_anomaly_score": "Score Anomalia Sweethearting",
      "cashier_sweethearting_z_score": "Z-Score Deviazione",
      "product_id": "Codice Prodotto",
      "weight": "Peso Rilevato",
      "line_total": "Totale Riga",
      "price_per_unit": "Prezzo per Unità",
      "is_anomaly": "Flag Anomalia"
    }
        
    integrity_payload = {
      "metadata": {
        "titolo": "Audit Integrità e Prevenzione Frodi",
        "descrizione": "Monitoraggio tecnico delle scansioni e della correttezza dei pesi alla vendita.",
        "arricchimento_business": "Strumento critico per la Loss Prevention. L'analisi incrociata tra anomalie di pesatura (ortofrutta) e punteggi di sweethearting permette di identificare perdite occulte che non emergono dai normali inventari. Consente interventi mirati di formazione o audit interno sui dipendenti con i profili di rischio più elevati.",
        "dizionario_dati": {
          "periodo": "L'intervallo temporale (anno-mese) a cui si riferisce l'analisi.",
          "Codice Cassiere": "Identificativo univoco dell'operatore sospettato di anomalie.",
          "Score Anomalia Sweethearting": "Valore binario o continuo derivato dal modello ML. Indica la probabilità di 'Sweethearting' (regalare merce scansionando prodotti a basso costo al posto di quelli costosi).",
          "Z-Score Deviazione": "Punteggio statistico che indica di quante deviazioni standard il cassiere è lontano dalla media. Sopra 2.0 è considerato un alert critico.",
          "Codice Prodotto": "Codice PLU o EAN dell'articolo ortofrutticolo che ha generato l'alert.",
          "Peso Rilevato": "Il peso in kg registrato dalla bilancia integrata nella cassa.",
          "Totale Riga": "Il prezzo finale calcolato per quel prodotto nella specifica transazione.",
          "Prezzo per Unità": "Il prezzo al kg (o per pezzo) applicato dal sistema.",
          "Flag Anomalia": "Indicatore booleano (Sì/No) che conferma se il rapporto peso/prezzo è fuori dai parametri di tolleranza impostati."
        }
      },
      "anomalie_sweethearting_cassieri": map_and_sanitize(global_cashier_sweethearting_anomalies, integrity_map),
      "alert_pesatura_ortofrutta": map_and_sanitize(global_produce_anomalies, integrity_map)
    }
    with open(json_integrity_path, 'w', encoding='utf-8') as f:
      json.dump(integrity_payload, f, indent=4, ensure_ascii=False, default=json_serial)
    logger.info(f"INFO: Exported Store Integrity Audit to {json_integrity_path}")
    
    # Export Behavioral Risk DNA (Fingerprints & Leakage)
    behavioral_map = {
      "cashier_id": "Codice Cassiere",
      "risk_score": "Score Rischio Comportamentale",
      "cash_reliance_ratio": "Indice Uso Contanti",
      "void_rate_per_receipt": "Tasso Storni",
      "total_markdowns": "Totale Sconti e Ribassi",
      "total_voids": "Totale Articoli Stornati",
      "captured_revenue": "Fatturato Acquisito",
      "leakage_pct": "Percentuale Perdita (Leakage)"
    }
        
    behavioral_payload = {
      "metadata": {
        "titolo": "DNA Comportamentale Cassieri e Radar Margine",
        "descrizione": "Profilazione del rischio operativo e analisi dell'erosione del margine per operatore.",
        "arricchimento_business": "Analizza l'efficienza operativa e la disciplina di cassa. Il calcolo della 'Percentuale Perdita (Leakage)' quantifica direttamente l'impatto economico degli errori o delle irregolarità procedurali. Questo report è fondamentale per identificare colli di bottiglia operativi e proteggere il margine netto del punto vendita attraverso il monitoraggio dei flussi di contante e storni.",
        "dizionario_dati": {
          "periodo": "L'intervallo temporale (anno-mese) a cui si riferisce l'analisi.",
          "Codice Cassiere": "Identificativo univoco del dipendente nel sistema POS.",
          "Score Rischio Comportamentale": "Indice sintetico (0-1) calcolato tramite algoritmi di clustering. Più è alto, più il comportamento del cassiere devia dalla norma del negozio.",
          "Indice Uso Contanti": "Rapporto tra transazioni in contanti e totale transazioni. Valori estremi possono indicare tentativi di manipolazione del fondo cassa.",
          "Tasso Storni": "Frequenza media di cancellazione di righe dallo scontrino. Un tasso alto è spesso correlato a merce consegnata ma non pagata (pass-outs).",
          "Totale Sconti e Ribassi": "Valore monetario totale degli sconti manuali applicati dal cassiere nel periodo.",
          "Totale Articoli Stornati": "Conteggio assoluto degli articoli cancellati dopo essere stati scansionati.",
          "Fatturato Acquisito": "Il volume totale di vendite nette (incassate correttamente) gestite dal cassiere.",
          "Percentuale Perdita (Leakage)": "Stima percentuale del fatturato perso a causa di anomalie operative (storni eccessivi, sconti non giustificati)."
        }
      },
      "fingerprint_comportamentale_cassieri": map_and_sanitize(global_cashier_fp_anomalies, behavioral_map),
      "analisi_perdita_margine": map_and_sanitize(global_margin_leakage, behavioral_map)
    }
    with open(json_behavioral_path, 'w', encoding='utf-8') as f:
      json.dump(behavioral_payload, f, indent=4, ensure_ascii=False, default=json_serial)
    logger.info(f"INFO: Exported Cashier Behavioral Risk DNA to {json_behavioral_path}")

  print(f"INFO: ⏱️ Total Execution Time: {time.time() - start_time:.2f}s")

if __name__ == "__main__":
  main()
