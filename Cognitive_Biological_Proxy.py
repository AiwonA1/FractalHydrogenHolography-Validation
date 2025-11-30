# HHF Cognitive / Biological Proxy Validation — Real Data Only (Console Output)
# ===========================================================================

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from urllib.request import urlopen
from io import StringIO

# -------------------------
# Fractal Dimension Functions
# -------------------------
def petrosian_fd(sig):
    sig = np.array(sig, dtype=float)
    N = len(sig)
    if N < 3:
        return np.nan
    diff = np.diff(sig)
    sign_changes = np.sum(diff[:-1] * diff[1:] < 0)
    if sign_changes <= 0:
        return np.nan
    return np.log10(N) / (np.log10(N) + np.log10(N / (N + 0.4 * sign_changes)))

def higuchi_fd(sig, k_max=10):
    L = []
    N = len(sig)
    for k in range(1, k_max + 1):
        Lk = []
        for m in range(k):
            Lmk = 0
            for i in range(1, int(np.floor((N - m) / k))):
                Lmk += abs(sig[m + i*k] - sig[m + (i-1)*k])
            Lmk = (Lmk * (N - 1) / (np.floor((N - m)/k) * k)) if Lmk != 0 else 0
            Lk.append(Lmk)
        L.append(np.mean(Lk))
    L = np.array(L)
    lnL = np.log(L[L>0])
    lnk = np.log(np.arange(1, len(L)+1))[L>0]
    if len(lnL) < 2:
        return np.nan
    return np.polyfit(lnk, lnL, 1)[0] * -1

# -------------------------
# Output Directories
# -------------------------
os.makedirs("plots_HHF", exist_ok=True)
os.makedirs("logs_HHF", exist_ok=True)

# -------------------------
# Real Datasets
# -------------------------
datasets_list = [
    {
        "name": "Daily_Minimum_Temperatures_Melbourne",
        "url": "https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv",
        "column": "Temp"
    },
    {
        "name": "InVitro_Microtubule_Oscillations",
        # Replace with real public microtubule dataset URL
        "url": "https://raw.githubusercontent.com/YourRepo/Microtubule_Oscillations.csv",
        "column": "Oscillation"
    }
]

results, log = [], {}

# -------------------------
# Process Datasets
# -------------------------
for ds in datasets_list:
    try:
        print(f"\nLoading {ds['name']} dataset...")
        response = urlopen(ds['url'])
        csv_data = response.read().decode('utf-8')
        df = pd.read_csv(StringIO(csv_data))
        sig = df[ds['column']].values
        
        # Fractal measures
        pfd = petrosian_fd(sig)
        hfd = higuchi_fd(sig)
        
        # Log and results
        log[ds['name']] = {"n_points": len(sig), "PFD": pfd, "HFD": hfd}
        results.append({"dataset": ds['name'], "n_points": len(sig), "PFD": pfd, "HFD": hfd})
        
        # Plot
        plt.figure(figsize=(12,4))
        plt.plot(sig)
        plt.title(f"{ds['name']} — PFD {pfd:.3f}, HFD {hfd:.3f}")
        plt.xlabel("Index")
        plt.ylabel("Signal")
        plt.tight_layout()
        plt.show()
        
        print(f"Processed {ds['name']}: PFD={pfd:.3f}, HFD={hfd:.3f}")
    
    except Exception as e:
        log[ds['name']] = {"error": str(e)}
        print(f"Error processing {ds['name']}: {e}")

# -------------------------
# Display Summary in Console
# -------------------------
df_summary = pd.DataFrame(results)
print("\n--- Fractal Analysis Summary ---")
print(df_summary.to_string(index=False))

print("\n--- Fractal Analysis Log (JSON) ---")
print(json.dumps(log, indent=2))

# -------------------------
# Save Files
# -------------------------
df_summary.to_csv("logs_HHF/fractal_summary.csv", index=False)
with open("logs_HHF/fractal_summary.json", "w") as f:
    json.dump(log, f, indent=2)

print("\nHHF Cognitive / Biological Proxy Validation Complete.")
print("All outputs (logs, plots, summary) displayed in console.")
