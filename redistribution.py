"""
redistribution.py
-----------------
Blood Transfer Recommendation Engine.

WHAT PROBLEM DOES THIS SOLVE?
  Some hospitals run low on blood while others have a surplus.
  This engine identifies those imbalances and recommends transfers —
  but only when the transfer makes clinical and logistical sense.

HOW ARE TRANSFERS SCORED?
  Each potential transfer gets a Priority Score:
    PS = W1 × ExpiryUrgency + W2 × TransportScore + W3 × ShortageSeverity
    W1=1.0  W2=0.5  W3=2.0

  - ExpiryUrgency:      1 / units_expiring_soon  (more urgent = higher score)
  - TransportScore:     1 / distance_km           (closer = higher score)
  - ShortageSeverity:   forecast - stock at recipient (bigger gap = higher score)

HARD CONSTRAINTS (a transfer is only recommended if ALL are satisfied):
  - Distance       ≤ 50 km
  - Transit time   ≤ 2 hours
  - Transfer qty   ≥ 10 units (too small to be worth the logistics)
  - Donor must retain enough stock for the next 3 days after donating

DAYS-OF-SUPPLY (DOS) THRESHOLDS:
  A hospital is a DONOR candidate only if it has > 8 days of supply.
  A hospital is a RECIPIENT candidate only if it has < 9 days of supply.
  (These two values overlap by 1 day intentionally — triggers a meaningful band.)
"""

import os, sys, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"; os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path: sys.path.insert(0, SRC_DIR)
from hybrid_ensemble import HybridForecaster

BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORTS_DIR = os.path.join(BASE_DIR, "outputs", "reports")
PLOTS_DIR   = os.path.join(BASE_DIR, "outputs", "plots")
os.makedirs(REPORTS_DIR, exist_ok=True); os.makedirs(PLOTS_DIR, exist_ok=True)


class RedistributionEngine:
    """
    Generates blood transfer recommendations across a 3-hospital network.
    All class-level constants are the operational parameters — easy to change.
    """

    # Hard operational constraints
    MAX_DIST     = 50.0   # max transfer distance in km
    MAX_TIME     = 2.0    # max transit time in hours
    MIN_QTY      = 10     # minimum units worth transferring
    SAFETY_DAYS  = 3      # donor must keep enough stock for this many days

    # Priority score weights (W3 is highest because shortage severity matters most)
    W1 = 1.0   # ExpiryUrgency weight
    W2 = 0.5   # TransportScore weight
    W3 = 2.0   # ShortageSeverity weight

    # Days-of-supply thresholds for triggering recommendations
    DONOR_DOS     = 8.0   # must have MORE than this to be a donor
    RECIPIENT_DOS = 9.0   # must have FEWER than this to be a recipient

    def __init__(self):
        """
        Load the hospital road network from CSV into a dictionary for fast lookups.
        The graph is bidirectional: A→B and B→A have the same distance/time.
        """
        g = pd.read_csv(os.path.join(BASE_DIR, "data", "raw", "hospital_graph.csv"))
        self.graph = {}
        for _, r in g.iterrows():
            info = {"distance_km": r["distance_km"], "transit_hr": r["transit_hr"]}
            self.graph.setdefault(r["from_hospital"], {})[r["to_hospital"]] = info
            self.graph.setdefault(r["to_hospital"],   {})[r["from_hospital"]] = info  # bidirectional

    def get_inventory_state(self, df, as_of_date):
        """
        Look up the most recent stock snapshot for each hospital-blood_group pair
        on or before the given date.
        Returns: {hospital: {blood_group: {stock, units_expiring_soon}}}
        """
        ao = pd.Timestamp(as_of_date)
        state = {}
        for (h, bg), g in df.groupby(["hospital", "blood_group"]):
            sub = g[g["date"] <= ao].sort_values("date")
            if sub.empty: continue
            lat = sub.iloc[-1]  # most recent row
            state.setdefault(h, {})[bg] = {
                "stock":               int(lat["stock_on_hand"]),
                "units_expiring_soon": int(lat["units_expiring_soon"]),
            }
        return state

    def compute_priority_score(self, donor, recipient, bg, stock, fc):
        """
        Calculate the Priority Score for a specific donor → recipient transfer.
        Higher score = more urgent / more beneficial transfer.
        """
        # ExpiryUrgency: 1 / units expiring soon at donor (urgent to move blood before it expires)
        eu = 1.0 / max(1, stock[donor][bg]["units_expiring_soon"])
        # TransportScore: 1 / distance (prefer nearby hospitals)
        ts = 1.0 / self.graph[donor][recipient]["distance_km"]
        # ShortageSeverity: how many units short the recipient will be vs forecast
        ss = max(0.0, fc.get(recipient, {}).get(bg, 0) - stock[recipient][bg]["stock"])
        return self.W1 * eu + self.W2 * ts + self.W3 * ss

    def generate_recommendations(self, df, forecast_dict, as_of_date):
        """
        Core recommendation logic.

        forecast_dict: {hospital: {blood_group: 7-day total forecast demand}}
        For each blood group, we check every possible donor → recipient pair,
        apply all constraints, compute the priority score, and collect valid transfers.
        The final list is sorted by priority score (highest first).
        """
        stock        = self.get_inventory_state(df, as_of_date)
        hospitals    = list(stock.keys())
        blood_groups = list(next(iter(stock.values())).keys())
        recs         = []

        for bg in blood_groups:
            # Convert 7-day forecast to daily rate (minimum 1 to avoid divide-by-zero)
            daily = {h: max(forecast_dict.get(h, {}).get(bg, 1) / 7.0, 1) for h in hospitals}

            for donor in hospitals:
                if bg not in stock.get(donor, {}): continue

                ds  = stock[donor][bg]["stock"]  # donor's current stock
                dd  = daily[donor]               # donor's daily demand rate
                dos = ds / dd                    # donor's days-of-supply

                # Skip donors that don't have enough stock to give
                if dos < self.DONOR_DOS: continue

                # Surplus = stock the donor has beyond its own 7-day forecast + safety buffer
                surplus = ds - forecast_dict.get(donor, {}).get(bg, 0) - self.SAFETY_DAYS * dd
                if surplus < self.MIN_QTY: continue  # not enough to transfer

                for recip in hospitals:
                    if recip == donor or bg not in stock.get(recip, {}): continue

                    # Check if this route exists and meets distance/time constraints
                    edge = self.graph.get(donor, {}).get(recip)
                    if not edge or edge["distance_km"] > self.MAX_DIST or edge["transit_hr"] > self.MAX_TIME:
                        continue

                    rs  = stock[recip][bg]["stock"]  # recipient's current stock
                    rd  = daily[recip]
                    # Skip recipients that already have enough supply
                    if rs / rd >= self.RECIPIENT_DOS: continue

                    # How many units does the recipient need to get back to RECIPIENT_DOS days?
                    deficit = max(
                        forecast_dict.get(recip, {}).get(bg, 0) - rs,  # forecast gap
                        int(self.RECIPIENT_DOS * rd) - rs,              # DOS-based gap
                        0
                    )

                    # Calculate transfer quantity
                    ex  = stock[donor][bg]["units_expiring_soon"]
                    qty = min(int(surplus), int(deficit))
                    if ex > 0: qty = max(qty, ex)   # always move expiring units if possible
                    # Final safety check: donor must keep 3-day buffer after transfer
                    if ds - qty < self.SAFETY_DAYS * dd:
                        qty = max(0, int(ds - self.SAFETY_DAYS * dd))
                    if qty < self.MIN_QTY: continue  # still not enough after adjustments

                    ps = self.compute_priority_score(donor, recip, bg, stock, forecast_dict)
                    recs.append({
                        "date":               pd.Timestamp(as_of_date).date().isoformat(),
                        "donor_hospital":     donor,
                        "recipient_hospital": recip,
                        "blood_group":        bg,
                        "transfer_qty":       int(qty),
                        "priority_score":     round(ps, 6),
                        "expiry_urgency":     round(self.W1 / max(1, ex), 6),
                        "transport_score":    round(self.W2 / edge["distance_km"], 6),
                        "shortage_severity":  round(self.W3 * max(0.0, forecast_dict.get(recip, {}).get(bg, 0) - rs), 6),
                        "distance_km":        edge["distance_km"],
                        "transit_hr":         edge["transit_hr"],
                        "donor_dos":          round(dos, 1),
                        "recipient_dos":      round(rs / rd, 1),
                        "status":             "PENDING",
                    })

        if not recs:
            # Return an empty DataFrame with the correct column structure
            return pd.DataFrame(columns=["date","donor_hospital","recipient_hospital","blood_group",
                "transfer_qty","priority_score","expiry_urgency","transport_score","shortage_severity",
                "distance_km","transit_hr","donor_dos","recipient_dos","status"])

        return pd.DataFrame(recs).sort_values("priority_score", ascending=False).reset_index(drop=True)

    def simulate_transfers(self, df, forecasters, start_date, end_date):
        """
        Roll through every day in the date range, generate recommendations,
        and auto-approve the top recommendation each day (for simulation purposes).

        In the real dashboard a human approves/rejects — this is just for bulk analysis.
        Returns: (all_recommendations_df, daily_summary_df)
        """
        all_recs, stats = [], []

        for date in pd.date_range(start=pd.Timestamp(start_date), end=pd.Timestamp(end_date), freq="D"):
            # Build a forecast for every hospital-blood_group on this date
            fcd = {}
            for (h, bg), hf in forecasters.items():
                try:
                    fcd.setdefault(h, {})[bg] = float(hf.predict(df, target_date=date)["hybrid_pred"].sum())
                except:
                    fcd.setdefault(h, {})[bg] = 0.0

            recs = self.generate_recommendations(df, fcd, date)

            if not recs.empty:
                # Auto-approve the highest-priority recommendation for simulation
                top = recs.iloc[0].copy(); top["status"] = "APPROVED_SIM"
                all_recs.append(top.to_dict())
                stats.append({
                    "date":                  date.date().isoformat(),
                    "n_recommendations":     len(recs),
                    "top_priority_score":    round(recs.iloc[0]["priority_score"], 4),
                    "total_qty_recommended": int(recs["transfer_qty"].sum()),
                })

        return (pd.DataFrame(all_recs) if all_recs else pd.DataFrame(),
                pd.DataFrame(stats)    if stats    else pd.DataFrame())


# ----------------------------------------------------------------------
# Run this file directly to simulate 3 years of transfers
# ----------------------------------------------------------------------
if __name__ == "__main__":
    df = pd.read_csv(os.path.join(BASE_DIR, "data", "raw", "synthetic_demand.csv"), parse_dates=["date"])
    combos = [("Hospital_A","O_positive"), ("Hospital_A","O_negative"),
              ("Hospital_B","O_positive"), ("Hospital_B","O_negative"),
              ("Hospital_C","O_positive"), ("Hospital_C","O_negative")]

    print("Loading hybrid forecasters...")
    fcs = {}
    for h, bg in combos:
        hf = HybridForecaster(h, bg); hf.load_models(); hf.optimise_weights(df)
        fcs[(h, bg)] = hf

    engine = RedistributionEngine()
    print(f"Simulating transfers: {df['date'].min().date()} to {df['date'].max().date()}")
    recs, _ = engine.simulate_transfers(df, fcs, df["date"].min(), df["date"].max())

    if not recs.empty:
        recs.to_csv(os.path.join(REPORTS_DIR, "transfer_recommendations.csv"), index=False)
        print(f"\nTransfer days: {len(recs)} | Total units: {int(recs['transfer_qty'].sum())}")
        print(f"All constraints satisfied:")
        print(f"  distance <= 50km : {(recs['distance_km'] <= 50).all()}")
        print(f"  qty >= 10 units  : {(recs['transfer_qty'] >= 10).all()}")
        print(recs[["date","donor_hospital","recipient_hospital","blood_group",
                    "transfer_qty","priority_score","donor_dos","recipient_dos"]].head(10).to_string(index=False))
    else:
        print("No transfer recommendations found in this period.")

    print("\nMilestone 7 complete.")
