"""
lstm_model.py
-------------
LSTM (Long Short-Term Memory) Neural Network Forecaster.

WHY LSTM?
  Blood demand is a time series — today's demand is related to yesterday's.
  LSTM is a type of Recurrent Neural Network specifically designed to learn
  patterns in sequential data.  It has "memory cells" that retain information
  over long sequences, making it much better than plain regression for this.

HOW IT WORKS (in simple terms):
  - Input:  the last 30 days of 8 features (demand, stock, weekday flags, etc.)
  - Output: the next 7 days of demand — all at once (multi-step forecasting)
  - Architecture: LSTM(50) → LSTM(50) → Dense(25) → Dense(7)
  - Training: minimise Mean Squared Error on an 80/20 train/validation split

The model is saved as a .keras file so it can be loaded later by the dashboard.
"""

import os, warnings, numpy as np, pandas as pd, joblib
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"          # silence TensorFlow startup messages
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf; tf.get_logger().setLevel("ERROR")
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler  # scales all values to [0, 1] range

# Folder paths
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LSTM_DIR    = os.path.join(BASE_DIR, "models", "lstm")
PLOTS_DIR   = os.path.join(BASE_DIR, "outputs", "plots")
REPORTS_DIR = os.path.join(BASE_DIR, "outputs", "reports")
for d in (LSTM_DIR, PLOTS_DIR, REPORTS_DIR): os.makedirs(d, exist_ok=True)

# The 8 columns the LSTM uses as input features.
# Demand is feature index 0 — important for the de-scaling step later.
FEATURE_COLS = [
    "demand",             # what we want to predict
    "stock_on_hand",      # current inventory level
    "is_weekend",         # binary: 1 if Saturday or Sunday
    "is_holiday",         # binary: 1 if Indian public holiday
    "is_dengue_season",   # binary: 1 if July–October
    "day_of_week",        # 0=Monday … 6=Sunday
    "month",              # 1–12
    "units_expiring_soon",# how many units expire within 3 days
]


def _mape(actual, predicted):
    """Mean Absolute Percentage Error — skip near-zero actual values to avoid huge errors."""
    mask = actual > 1
    return float(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100) if mask.sum() else 0.0


class LSTMForecaster:
    """
    Trains a two-layer LSTM network to predict the next 7 days of blood demand
    from a 30-day sliding window of 8 input features.
    """

    def __init__(self, hospital, blood_group, window=30, horizon=7):
        self.hospital, self.blood_group = hospital, blood_group
        self.window  = window   # how many past days to feed into the LSTM
        self.horizon = horizon  # how many future days to predict
        self.model_path  = os.path.join(LSTM_DIR, f"{hospital}_{blood_group}_lstm.keras")
        self.scaler_path = os.path.join(LSTM_DIR, f"{hospital}_{blood_group}_scaler.pkl")
        self._model = self._scaler = None

    def build_sequences(self, data, window, horizon):
        """
        Convert a 2D array (days × features) into supervised learning examples.

        Sliding window approach:
          X[i] = data[i : i+window]        → the 30-day input
          y[i] = data[i+window : i+window+horizon, 0]  → next 7 days of demand (col 0)

        Example with window=3, horizon=2, data = [a, b, c, d, e, f]:
          X[0] = [a, b, c]   y[0] = [d, e]
          X[1] = [b, c, d]   y[1] = [e, f]
        """
        X, y = [], []
        for i in range(len(data) - window - horizon + 1):
            X.append(data[i : i + window])
            y.append(data[i + window : i + window + horizon, 0])  # column 0 = demand
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

    def build_model(self):
        """
        Define the neural network architecture:
          Input   → (30 days × 8 features)
          LSTM(50)  → first LSTM layer; return_sequences=True passes its output to the next LSTM
          LSTM(50)  → second LSTM layer; distils the sequence into a single context vector
          Dense(25) → fully-connected layer to learn non-linear combinations
          Dense(7)  → final output: 7-day demand forecast
        """
        m = Sequential([
            Input(shape=(self.window, len(FEATURE_COLS))),
            LSTM(50, return_sequences=True,  activation="tanh"),  # 50 memory units
            LSTM(50, return_sequences=False, activation="tanh"),
            Dense(25, activation="relu"),    # ReLU avoids the vanishing gradient problem
            Dense(self.horizon, activation="relu"),  # output must be >= 0
        ])
        m.compile(optimizer=Adam(0.001), loss="mse")  # Adam optimizer + Mean Squared Error loss
        return m

    def train(self, df):
        """
        Train the LSTM model on historical data for this hospital + blood group.

        Process:
        1. Filter data, sort by date
        2. Scale all features to [0,1] using MinMaxScaler (neural nets train better on small numbers)
        3. Split 80% train / 20% validation (chronologically — no shuffling for time series!)
        4. Build sliding-window sequences
        5. Train with EarlyStopping (stops if validation loss stops improving)
        6. Save the model and scaler to disk
        """
        sub = df[(df["hospital"] == self.hospital) & (df["blood_group"] == self.blood_group)].sort_values("date").reset_index(drop=True)
        raw = sub[FEATURE_COLS].values.astype(np.float32)
        sp  = int(len(raw) * 0.8)  # 80/20 split point

        # Fit the scaler on training data ONLY — never let validation data influence the scaler
        self._scaler = MinMaxScaler()
        tr = self._scaler.fit_transform(raw[:sp])
        vl = self._scaler.transform(raw[sp:])

        Xtr, ytr = self.build_sequences(tr, self.window, self.horizon)
        Xvl, yvl = self.build_sequences(vl, self.window, self.horizon)

        self._model = self.build_model()
        hist = self._model.fit(
            Xtr, ytr, validation_data=(Xvl, yvl),
            epochs=100, batch_size=32,
            callbacks=[
                # Stop training if val_loss hasn't improved for 10 epochs (saves time)
                EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True, verbose=0),
                # Always save the best version of the model seen so far
                ModelCheckpoint(self.model_path, monitor="val_loss", save_best_only=True, verbose=0),
            ],
            verbose=0,
        )
        self._model.save(self.model_path)
        joblib.dump(self._scaler, self.scaler_path)
        print(f"  val_loss={hist.history['val_loss'][-1]:.4f} ({len(hist.history['loss'])} epochs)", end="")
        return hist

    def _descale(self, scaled_demand):
        """
        Convert scaled predictions back to real unit counts.
        MinMaxScaler needs a full row (all 8 features) to inverse-transform.
        We put our scaled demand in column 0 and zeros elsewhere, then read column 0 back.
        """
        dummy = np.zeros((len(scaled_demand), len(FEATURE_COLS)), dtype=np.float32)
        dummy[:, 0] = scaled_demand  # put demand in the first column
        return self._scaler.inverse_transform(dummy)[:, 0]

    def predict(self, recent_df):
        """
        Predict the next 7 days given the last 30 days of data.
        recent_df must contain at least 30 rows for this hospital + blood group.
        """
        sub    = recent_df[(recent_df["hospital"] == self.hospital) & (recent_df["blood_group"] == self.blood_group)].sort_values("date").tail(self.window)
        scaled = self._scaler.transform(sub[FEATURE_COLS].values.astype(np.float32))
        # Add a batch dimension: shape (1, 30, 8) — the model expects batches
        pred_scaled = self._model.predict(scaled[np.newaxis, :, :], verbose=0)[0]
        return self._descale(pred_scaled).clip(0)  # demand can't be negative

    def evaluate(self, df):
        """
        Measure model accuracy on the last 90 days (hold-out period).
        Uses rolling 7-day predictions, then compares to actual demand.
        """
        sub    = df[(df["hospital"] == self.hospital) & (df["blood_group"] == self.blood_group)].sort_values("date").reset_index(drop=True)
        scaled = self._scaler.transform(sub[FEATURE_COLS].values.astype(np.float32))

        Xa, ya = self.build_sequences(scaled, self.window, self.horizon)
        # Index of the first window whose target falls within the last 90 days
        fh = len(Xa) - (90 // self.horizon)

        ps = self._model.predict(Xa[fh:], verbose=0)
        preds  = np.concatenate([self._descale(p) for p in ps]).clip(0)
        actual = np.concatenate([self._descale(y) for y in ya[fh:]])

        return {
            "MAE":  round(float(np.mean(np.abs(actual - preds))), 3),
            "RMSE": round(float(np.sqrt(np.mean((actual - preds) ** 2))), 3),
            "MAPE": round(_mape(actual, preds), 3),
        }

    def load(self):
        """Load a previously saved model and scaler from disk."""
        self._model  = load_model(self.model_path)
        self._scaler = joblib.load(self.scaler_path)


# ----------------------------------------------------------------------
# Run this file directly to train all 6 LSTM models
# ----------------------------------------------------------------------
if __name__ == "__main__":
    df = pd.read_csv(os.path.join(BASE_DIR, "data", "raw", "synthetic_demand.csv"), parse_dates=["date"])
    combos = [("Hospital_A","O_positive"), ("Hospital_A","O_negative"),
              ("Hospital_B","O_positive"), ("Hospital_B","O_negative"),
              ("Hospital_C","O_positive"), ("Hospital_C","O_negative")]

    rows, histories = [], {}
    for h, bg in combos:
        print(f"  {h} {bg}...", end="", flush=True)
        lf   = LSTMForecaster(h, bg)
        hist = lf.train(df)
        histories[(h, bg)] = hist
        m = lf.evaluate(df)
        rows.append({"hospital": h, "blood_group": bg, **m})
        print(f"  MAE={m['MAE']:.2f}  RMSE={m['RMSE']:.2f}  MAPE={m['MAPE']:.2f}%")

    pd.DataFrame(rows).to_csv(os.path.join(REPORTS_DIR, "lstm_metrics.csv"), index=False)

    # Plot training loss curve for Hospital_A O_positive
    hist_a = histories[("Hospital_A", "O_positive")]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(hist_a.history["loss"],     label="Train Loss", color="steelblue")
    ax.plot(hist_a.history["val_loss"], label="Val Loss",   color="orange", linestyle="--")
    ax.set(title="LSTM Training Loss — Hospital_A O_positive", xlabel="Epoch", ylabel="MSE Loss")
    ax.legend(); plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "lstm_training_curve_HospA_Opos.png"), dpi=100); plt.close()
    print(f"\nMetrics + training curve saved.")
