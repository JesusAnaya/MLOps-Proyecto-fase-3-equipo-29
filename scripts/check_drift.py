import pandas as pd
import os
import joblib
import json

from evidently import Report
from evidently.presets import DataDriftPreset

BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
REFERENCE_DATA_FILE = os.path.join(BASE_DIR, 'data', 'processed', 'data_clean.csv')
CURRENT_DATA_FILE = os.path.join(BASE_DIR, 'data', 'processed', 'drift_south_test_data.csv')
REPORT_PATH = os.path.join(BASE_DIR, 'reports')
REPORT_JSON = 'evidently_drift_results.json'
REPORT_HTML = 'evidently_drift_report.html'
MODEL_FILE = os.path.join(BASE_DIR, 'models', 'best_model.joblib')

TARGET_NAME = 'kredit'
PREDICTION_NAME = 'prediction'


def load_data(path):
    return pd.read_csv(path)


def run_evidently_report(ref_data, cur_data):
    os.makedirs(REPORT_PATH, exist_ok=True)

    # Crear el reporte con DataDriftPreset
    report = Report(metrics=[DataDriftPreset()])
    eval_result = report.run(reference_data=ref_data, current_data=cur_data)

    # Guardar JSON completo
    json_path = os.path.join(REPORT_PATH, REPORT_JSON)
    with open(json_path, "w") as f:
        f.write(eval_result.json())

    # Crear HTML manualmente
    html_path = os.path.join(REPORT_PATH, REPORT_HTML)
    report_list = eval_result.dict()["metrics"] 
    html_content = "<html><body><h1>Reporte Evidently – Data Drift</h1>"

    for metric in report_list:
        metric_name = metric.get("metric", "Unknown")
        metric_result = metric.get("result", {})
        html_content += f"<h2>{metric_name}</h2><pre>{json.dumps(metric_result, indent=4)}</pre>"

    html_content += "</body></html>"

    with open(html_path, "w") as f:
        f.write(html_content)

    return html_path, json_path


if __name__ == "__main__":
    ref = load_data(REFERENCE_DATA_FILE)
    cur = load_data(CURRENT_DATA_FILE)

    model = joblib.load(MODEL_FILE)

    feature_cols = ref.drop(columns=[TARGET_NAME], errors="ignore").columns
    ref[PREDICTION_NAME] = model.predict(ref[feature_cols])
    cur[PREDICTION_NAME] = model.predict(cur[feature_cols])

    html, js = run_evidently_report(ref, cur)
    print("Saved HTML:", html)
    print("Saved JSON:", js)