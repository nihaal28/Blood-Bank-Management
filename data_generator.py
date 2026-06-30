"""
data_generator.py
-----------------
Creates a SYNTHETIC (fake but realistic) blood demand dataset for 3 hospitals
over 3 years (2021-2023). Because real patient data is private, we simulate it
using statistical distributions that mimic real blood bank behaviour.

Output: data/raw/synthetic_demand.csv  (6,570 rows × 17 columns)
        data/raw/hospital_graph.csv    (distances between hospitals)
"""

import os, numpy as np, pandas as pd
from datetime import date, timedelta

# Figure out where the project root folder is, then set up the output folder
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR  = os.path.join(BASE_DIR, "data", "raw")
os.makedirs(RAW_DIR, exist_ok=True)  # create folder if it doesn't exist yet

# Indian public holidays — demand drops on these days because elective surgeries
# are usually cancelled, so blood usage falls by ~45%
HOLIDAYS = {
    date(2021,1,26), date(2021,3,29), date(2021,8,15), date(2021,10,2),  date(2021,11,4),  date(2021,12,25),
    date(2022,1,26), date(2022,3,18), date(2022,8,15), date(2022,10,2),  date(2022,10,24), date(2022,12,25),
    date(2023,1,26), date(2023,3,8),  date(2023,8,15), date(2023,10,2),  date(2023,11,12), date(2023,12,25),
}

# Average daily blood demand (lambda for Poisson distribution) per hospital & blood type.
# Hospital_A is a large city hospital, Hospital_C is a small clinic — hence different values.
BASE_LAMBDA = {
    ("Hospital_A", "O_positive"): 45,  # ~45 units/day on average
    ("Hospital_A", "O_negative"): 18,
    ("Hospital_B", "O_positive"): 28,
    ("Hospital_B", "O_negative"): 11,
    ("Hospital_C", "O_positive"): 14,
    ("Hospital_C", "O_negative"):  6,
}


class SyntheticBloodDataGenerator:
    """
    Generates a realistic synthetic blood demand & inventory dataset.
    Uses a Poisson distribution for daily demand (common in healthcare modelling)
    and a FIFO (First-In-First-Out) simulation for inventory tracking.
    """

    def __init__(self, seed=42):
        # seed=42 means the "random" numbers are the same every run → reproducible results
        self.rng = np.random.default_rng(seed)
        self.hospitals    = ["Hospital_A", "Hospital_B", "Hospital_C"]
        self.blood_groups = ["O_positive", "O_negative"]
        self.start, self.end = date(2021, 1, 1), date(2023, 12, 31)

    # ------------------------------------------------------------------
    # DEMAND SIMULATION
    # ------------------------------------------------------------------
    def _demand(self, h, bg, d):
        """
        Simulate how many blood units are needed on a single day.

        We start with the average (lambda) and apply real-world multipliers:
        - Weekends: fewer surgeries → 30% less demand
        - Mondays: post-weekend surgical backlog → 20% more demand
        - Public holidays: elective procedures cancelled → 45% less demand
        - Dengue season (Jul-Oct): more O+ blood needed for patients
        - Random trauma events: 5% chance each day of a large emergency spike
        """
        lam = float(BASE_LAMBDA[(h, bg)])  # start with the average

        # Apply day-of-week adjustments (weekday() 0=Mon, 5=Sat, 6=Sun)
        if d.weekday() in (5, 6):  lam *= 0.70   # weekend dip
        elif d.weekday() == 0:     lam *= 1.20   # Monday surgical surge

        # Holiday dip (applied after weekday adjustments)
        if d in HOLIDAYS: lam *= 0.55

        # Draw a random demand from the Poisson distribution
        # Poisson is ideal here: counts of events (blood requests) per time period
        v = int(self.rng.poisson(lam))

        # Dengue season spike — only affects O_positive blood type
        if bg == "O_positive" and 7 <= d.month <= 10:
            v += int(self.rng.poisson(8))  # extra ~8 units/day during monsoon

        # Random trauma event (road accident, mass casualty) — 5% daily probability
        if self.rng.random() < 0.05:
            v += int(self.rng.poisson(12))  # sudden spike of ~12 extra units

        return max(0, v)  # demand can never be negative

    # ------------------------------------------------------------------
    # INVENTORY SIMULATION (FIFO)
    # ------------------------------------------------------------------
    def _inventory(self, demands, dates):
        """
        Simulate the blood bank inventory day by day using FIFO logic.

        FIFO (First-In-First-Out) = oldest blood units are used first,
        which is exactly how real blood banks operate to minimise wastage.

        Each 'batch' is represented as [expiry_date, quantity].
        Every day we:
          1. Remove expired batches
          2. Add newly collected blood (1.1× demand so we have a small surplus)
          3. Fulfil demand by consuming the oldest batches first
          4. Record what happened (stock, used, expired, shortages)
        """
        # Seed the initial stock at the start of the simulation
        # Use 0.8-2.5× the first week's demand, spread across 5 batches
        init = int(self.rng.uniform(0.8, 2.5) * (sum(demands[:7]) if len(demands) >= 7 else sum(demands)))
        batches = [
            [dates[0] + timedelta(days=int(self.rng.integers(5, 35))), init // 5]
            for _ in range(5)
        ]

        # Lists to collect daily stats — one value per day
        soh, col, used, exp, soon, short = [], [], [], [], [], []

        for d, dem in zip(dates, demands):

            # Step 1: Remove batches that expired before today
            expired = sum(q for e, q in batches if e < d)
            batches  = [[e, q] for e, q in batches if e >= d]

            # Step 2: Collect new blood — 1.1× demand ensures a ~10% surplus
            # which produces a realistic discard rate (some blood will expire unused)
            c = int(self.rng.poisson(max(1, dem * 1.1)))
            batches.append([d + timedelta(days=int(self.rng.integers(20, 43))), c])

            # Step 3: Record total stock BEFORE consuming today's demand
            stock = sum(q for _, q in batches)

            # Step 4: Consume demand using FIFO — oldest batches consumed first
            rem, nb = dem, []  # rem = remaining demand still to be filled
            for e, q in sorted(batches, key=lambda x: x[0]):  # sorted by expiry (oldest first)
                if rem <= 0:    nb.append([e, q])         # demand met, keep this batch
                elif q <= rem:  rem -= q                   # this batch fully consumed
                else:           nb.append([e, q - rem]); rem = 0  # partially consumed
            batches = nb

            # Step 5: Units expiring within the next 3 days (an early warning signal)
            thr = d + timedelta(days=3)

            # Record all stats for this day
            soh.append(sum(q for _, q in batches))   # stock remaining after consumption
            col.append(c)                              # how much was collected today
            used.append(dem - max(0, rem))             # how much was actually used
            exp.append(expired)                        # how much expired today
            soon.append(sum(q for e, q in batches if e <= thr))  # soon-to-expire
            short.append(max(0, dem - stock))          # unmet demand = shortage

        return dict(stock_on_hand=soh, units_collected=col, units_used=used,
                    units_expired=exp, units_expiring_soon=soon, shortage=short)

    # ------------------------------------------------------------------
    # MAIN GENERATION METHOD
    # ------------------------------------------------------------------
    def generate(self):
        """
        Run demand + inventory simulation for all 6 hospital-blood_group combinations
        and assemble everything into a single DataFrame.
        """
        # Build a list of every date from 2021-01-01 to 2023-12-31
        all_dates = [self.start + timedelta(days=i) for i in range((self.end - self.start).days + 1)]

        rows = []
        for h in self.hospitals:
            for bg in self.blood_groups:
                print(f"  {h} {bg}...", end="", flush=True)

                # Simulate demand for every day, then run the inventory model
                dem = [self._demand(h, bg, d) for d in all_dates]
                inv = self._inventory(dem, all_dates)

                # Package every day into a row with all 17 columns
                for i, d in enumerate(all_dates):
                    rows.append({
                        "date":               d.isoformat(),
                        "hospital":           h,
                        "blood_group":        bg,
                        "demand":             dem[i],
                        "units_collected":    inv["units_collected"][i],
                        "stock_on_hand":      inv["stock_on_hand"][i],
                        "units_used":         inv["units_used"][i],
                        "units_expired":      inv["units_expired"][i],
                        "units_expiring_soon":inv["units_expiring_soon"][i],
                        "shortage":           inv["shortage"][i],
                        # Useful binary flags for the ML models
                        "is_weekend":         int(d.weekday() in (5, 6)),
                        "is_holiday":         int(d in HOLIDAYS),
                        "is_dengue_season":   int(7 <= d.month <= 10),
                        "day_of_week":        d.weekday(),   # 0=Mon … 6=Sun
                        "month":              d.month,
                        "year":               d.year,
                        "week_of_year":       d.isocalendar()[1],
                    })
                print(" done")

        df = pd.DataFrame(rows)
        df["date"] = pd.to_datetime(df["date"])  # convert string dates to datetime objects
        return df

    @staticmethod
    def save_hospital_graph():
        """
        Save the road-network between hospitals (distances + travel times).
        This is used by the redistribution engine to check feasibility of transfers.
        """
        graph = pd.DataFrame([
            {"from_hospital": "Hospital_A", "to_hospital": "Hospital_B", "distance_km": 18, "transit_hr": 0.5},
            {"from_hospital": "Hospital_A", "to_hospital": "Hospital_C", "distance_km": 44, "transit_hr": 1.2},
            {"from_hospital": "Hospital_B", "to_hospital": "Hospital_C", "distance_km": 31, "transit_hr": 0.9},
        ])
        graph.to_csv(os.path.join(RAW_DIR, "hospital_graph.csv"), index=False)
        return graph


# ----------------------------------------------------------------------
# Run this file directly to regenerate the dataset
# ----------------------------------------------------------------------
if __name__ == "__main__":
    gen  = SyntheticBloodDataGenerator(seed=42)
    df   = gen.generate()
    path = os.path.join(RAW_DIR, "synthetic_demand.csv")
    df.to_csv(path, index=False)
    gen.save_hospital_graph()

    print(f"\nRows: {len(df)} | Range: {df['date'].min().date()} to {df['date'].max().date()}")
    print(df.groupby(["hospital", "blood_group"])["demand"].mean().round(1).to_string())
    print(f"\nSaved to {path}")
