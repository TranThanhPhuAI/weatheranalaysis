# app.py — model.pkl + fake_full_data.csv (+ lag/rolling + quantile clipping)
from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import logging, os, json

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# -----------------------------
# Paths
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
pjoin = lambda *x: os.path.join(BASE_DIR, *x)

MODEL_PKL   = pjoin("model.pkl")
WEATHER_CSV = pjoin("fake_full_data.csv")   # dữ liệu thời tiết bạn đang dùng
META_JSON   = pjoin("model_meta.json")      # tùy chọn (quantile clip); có cũng được, không có vẫn chạy

# -----------------------------
# Load model
# -----------------------------
try:
    import joblib
    _use_joblib = True
except Exception:
    import pickle
    _use_joblib = False

def load_model(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Không tìm thấy model: {path}")
    if _use_joblib:
        return joblib.load(path)
    with open(path, "rb") as f:
        return pickle.load(f)

model = load_model(MODEL_PKL)

# danh sách cột model mong đợi (nếu có)
try:
    MODEL_FEATURES = list(getattr(model, "feature_names_in_", None))
except Exception:
    MODEL_FEATURES = None

# -----------------------------
# Load meta (tùy chọn)
# -----------------------------
META = None
if os.path.exists(META_JSON):
    with open(META_JSON, "r", encoding="utf-8") as f:
        META = json.load(f)
    app.logger.info(f"✅ Loaded model_meta.json ({len(META.get('features', []))} features)")
else:
    app.logger.warning("⚠️  Không có model_meta.json -> bỏ qua quantile clipping")

# -----------------------------
# Load weather data
# -----------------------------
if not os.path.exists(WEATHER_CSV):
    raise FileNotFoundError("Không tìm thấy fake_full_data.csv trong thư mục dự án!")

df = pd.read_csv(WEATHER_CSV)
df.columns = [c.strip() for c in df.columns]
df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()

BASE_FEATS = ["PRCP","RHAV","TAVG","TMAX","TMIN","AWND","CDD","HDD"]
app.logger.info(f"✅ Weather file: {WEATHER_CSV} | rows={len(df)}")

# -----------------------------
# Helpers
# -----------------------------
def _safe(v, default=0.0):
    try:
        if pd.isna(v): return float(default)
        return float(v)
    except Exception:
        return float(default)

def _synthetic(date: pd.Timestamp):
    """Sinh thời tiết hợp lý theo tháng (fallback khi thiếu dòng thật)."""
    seed = int(date.strftime("%Y%m%d"))
    rng = np.random.default_rng(seed)
    m = date.month
    base_temp = {1:12,2:13,3:14.5,4:16.5,5:18.5,6:21,7:23,8:23,9:22,10:19,11:15,12:12.5}[m]
    TAVG = rng.normal(base_temp, 1.2)
    return {
        "PRCP": max(0.0, rng.exponential(1.0) - 0.25),
        "RHAV": float(np.clip(rng.normal(65, 10), 40, 95)),
        "TAVG": TAVG,
        "TMAX": TAVG + rng.uniform(2, 4),
        "TMIN": TAVG - rng.uniform(2, 4),
        "AWND": rng.uniform(1, 5),        # m/s
        "CDD":  max(0, TAVG - 18.3),
        "HDD":  max(0, 18.3 - TAVG),
    }

def _compute_lag_roll(date: pd.Timestamp):
    """
    Tạo *_lag1 & *_roll7 từ lịch sử t-1 trong df (nếu model cần).
    Nếu không đủ dữ liệu -> trả 0.0 để app vẫn chạy.
    """
    d1 = date - pd.Timedelta(days=1)
    hist = df[df["date"] <= d1].sort_values("date")
    last7 = hist.tail(7).copy()

    out = {}
    if not last7.empty:
        last = last7.iloc[-1]
        for c in BASE_FEATS:
            out[f"{c}_lag1"]  = _safe(last.get(c), 0.0)
            out[f"{c}_roll7"] = _safe(last7[c].mean(), 0.0)
    else:
        for c in BASE_FEATS:
            out[f"{c}_lag1"]  = 0.0
            out[f"{c}_roll7"] = 0.0

    # nếu model dùng demand_lag1, set ở đây (không có data điện t-1 thì để 0.0)
    out["demand_lag1"] = 0.0
    return out

def _clip_by_meta(vals: dict):
    """Giới hạn mỗi feature theo [p01, p99] của tập train (nếu có meta)."""
    if not META: 
        return vals
    q = META.get("quantiles", {})
    out = vals.copy()
    for k, v in list(vals.items()):
        if k in q:
            lo = q[k]["p01"]; hi = q[k]["p99"]
            out[k] = float(np.clip(v, lo, hi))
    return out

def _log_debug(tag, d: dict):
    try:
        pairs = []
        for k in sorted(d.keys()):
            v = d[k]
            if isinstance(v, (int, float, np.floating)):
                pairs.append(f"{k}={float(v):.3f}")
        app.logger.info(f"{tag}: " + " | ".join(pairs))
    except Exception:
        pass

def _build_X(vals: dict):
    cols = MODEL_FEATURES if (MODEL_FEATURES and len(MODEL_FEATURES) > 0) else BASE_FEATS
    row = [float(vals.get(c, 0.0)) for c in cols]
    X = np.array([row], dtype=float)
    # đảm bảo đúng kích thước model yêu cầu
    try:
        need = int(getattr(model, "n_features_in_", X.shape[1]))
        if X.shape[1] < need:
            X = np.pad(X, ((0,0),(0, need - X.shape[1])), mode="constant")
        elif X.shape[1] > need:
            X = X[:, :need]
    except Exception:
        pass
    return X, cols

# -----------------------------
# Routes
# -----------------------------
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(silent=True) or {}
        date_str = data.get("date")
        if not date_str:
            return jsonify({"error": "Không có ngày nào được chọn!"}), 400

        date = pd.to_datetime(date_str, errors="coerce")
        if pd.isna(date):
            return jsonify({"error": "Định dạng ngày không hợp lệ!"}), 400
        date = pd.Timestamp(date).normalize()

        # lấy thời tiết thật nếu có, không thì synthetic
        row = df.loc[df["date"] == date]
        if row.empty:
            vals = _synthetic(date)
            source = "synthetic"
        else:
            s = row.iloc[0]
            vals = {
                "PRCP": _safe(s.get("PRCP", 0)),
                "RHAV": _safe(s.get("RHAV", 60)),
                "TAVG": _safe(s.get("TAVG", 18)),
                "TMAX": _safe(s.get("TMAX", 20)),
                "TMIN": _safe(s.get("TMIN", 16)),
                "AWND": _safe(s.get("AWND", 2.5)),
                "CDD":  _safe(s.get("CDD", 0)),
                "HDD":  _safe(s.get("HDD", 0)),
            }
            source = "observed"

        # bổ sung lag/roll nếu model cần (không ảnh hưởng nếu model không dùng các cột này)
        vals.update(_compute_lag_roll(date))

        # clip theo phân bố train (nếu có meta) và log giá trị feed vào model
        vals = _clip_by_meta(vals)
        _log_debug("FEED_TO_MODEL", vals)

        # build X & predict
        X, used_cols = _build_X(vals)
        yhat = float(model.predict(X)[0])
        wind_kmh = vals["AWND"] * 3.6  # m/s -> km/h (chỉ để hiển thị UI)

        return jsonify({
            "date": date.strftime("%Y-%m-%d"),
            "prediction": round(yhat, 2),
            "temperature": round(vals["TAVG"], 1),
            "humidity": round(vals["RHAV"], 1),
            "rainfall": round(vals["PRCP"], 1),
            "wind_speed": round(wind_kmh, 1),
            "CDD": round(vals["CDD"], 1),
            "HDD": round(vals["HDD"], 1),
            "meta": {"source": source, "used_columns": used_cols}
        })

    except Exception as e:
        app.logger.exception("Prediction error")
        return jsonify({"error": f"Lỗi máy chủ: {str(e)}"}), 500

# -----------------------------
# Run
# -----------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8501, debug=True)
