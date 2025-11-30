# ================================================================
# HHF Cognitive / Biological Proxy Validation
# Dataset: Daily Minimum Temperatures — Melbourne
# Outputs: Console logs, plots, summary report
# ================================================================

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------
# Fractal Dimension Functions
# -------------------------
def petrosian_fd(sig):
    """Compute Petrosian Fractal Dimension (PFD) of a signal."""
    sig = np.array(sig, dtype=float)
    N = len(sig)
    if N < 3:
        return np.nan
    diff = np.diff(sig)
    sign_changes = np.sum(diff[:-1]*diff[1:] < 0)
    if sign_changes <= 0:
        return np.nan
    return np.log10(N)/(np.log10(N) + np.log10(N/(N + 0.4*sign_changes)))

def higuchi_fd(sig, k_max=10):
    """Compute Higuchi Fractal Dimension (HFD) of a signal."""
    sig = np.array(sig, dtype=float)
    N = len(sig)
    L = []
    x = sig
    for k in range(1, k_max+1):
        Lk = []
        for m in range(k):
            Lmk = np.sum(np.abs(x[m+k:N:k] - x[m+k-1:N-k:k]))
            norm = (N-1)//k * k
            Lmk = (Lmk * (N-1)/norm)/k
            Lk.append(Lmk)
        L.append(np.mean(Lk))
    lnL = np.log(L)
    lnk = np.log(1./np.arange(1,k_max+1))
    # Linear fit
    A = np.vstack([lnk, np.ones(len(lnk))]).T
    fd, _ = np.linalg.lstsq(A, lnL, rcond=None)[0]
    return fd

# -------------------------
# Output Directories
# -------------------------
os.makedirs("plots_temperature", exist_ok=True)
os.makedirs("logs_temperature", exist_ok=True)

# -------------------------
# Load Dataset
# -------------------------
print("Loading Daily Minimum Temperatures (Melbourne) dataset...")
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv"

try:
    df_temp = pd.read_csv(url)
    sig = df_temp['Temp'].values
    n_points = len(sig)
    
    # Compute fractal dimensions
    pfd_val = petrosian_fd(sig)
    hfd_val = higuchi_fd(sig)
    
    # Log results
    results = {
        "Daily_Minimum_Temperatures_Melbourne": {
            "n_points": n_points,
            "PFD": pfd_val,
            "HFD": hfd_val
        }
    }
    
    # Plot
    plt.figure(figsize=(12,4))
    plt.plot(sig, color='tab:blue')
    plt.title(f"Daily Minimum Temperatures — PFD {pfd_val:.3f}, HFD {hfd_val:.3f}")
    plt.xlabel("Day Index")
    plt.ylabel("Temperature (°C)")
    plt.tight_layout()
    plt.savefig("plots_temperature/Daily_Min_Temperatures.png")
    plt.show()
    
    # Summary DataFrame
    df_summary = pd.DataFrame({
        "dataset": ["Daily_Minimum_Temperatures_Melbourne"],
        "n_points": [n_points],
        "PFD": [pfd_val],
        "HFD": [hfd_val]
    })
    
    # Save logs
    with open("logs_temperature/fractal_analysis_log.json","w") as f:
        json.dump(results,f,indent=2)
    df_summary.to_csv("logs_temperature/fractal_analysis_summary.csv", index=False)
    
    # Console Output
    print(f"\nProcessed Daily_Minimum_Temperatures_Melbourne: PFD={pfd_val:.3f}, HFD={hfd_val:.3f}")
    print("\n--- Fractal Analysis Summary ---")
    print(df_summary)
    print("\n--- Fractal Analysis Log (JSON) ---")
    print(json.dumps(results, indent=2))
    print("\nHHF Cognitive / Biological Proxy Validation — Daily Temperature Time-Series Complete.")
    print("All outputs (logs, plots, summary) displayed in console.")
    
except Exception as e:
    print("Error loading or processing dataset:", e)
