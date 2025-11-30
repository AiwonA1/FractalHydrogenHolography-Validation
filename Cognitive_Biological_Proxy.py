# ======================================================
# HHF Cognitive / Biological Proxy Validation Pipeline
# Includes microtubule oscillation simulation component
# ======================================================

import os, json, numpy as np, pandas as pd, matplotlib.pyplot as plt

# -------------------------
# Fractal Measures
# -------------------------
def petrosian_fd(sig):
    """Petrosian Fractal Dimension"""
    sig = np.array(sig, dtype=float)
    N = len(sig)
    if N < 3: return np.nan
    diff = np.diff(sig)
    sign_changes = np.sum(diff[:-1]*diff[1:] < 0)
    if sign_changes <= 0: return np.nan
    return np.log10(N)/(np.log10(N) + np.log10(N/(N+0.4*sign_changes)))

def higuchi_fd(sig, kmax=10):
    """Higuchi Fractal Dimension"""
    L = []
    N = len(sig)
    for k in range(1, kmax+1):
        Lk = []
        for m in range(k):
            Lm = 0
            for i in range(1, int(np.floor((N-m)/k))):
                Lm += abs(sig[m+i*k] - sig[m+(i-1)*k])
            Lm = Lm*(N-1)/(np.floor((N-m)/k)*k)
            Lk.append(Lm)
        L.append(np.mean(Lk))
    lnL = np.log(L)
    lnk = np.log(1./np.arange(1, kmax+1))
    return np.polyfit(lnk, lnL, 1)[0]

# -------------------------
# Directories
# -------------------------
os.makedirs("plots_hhf", exist_ok=True)
os.makedirs("logs_hhf", exist_ok=True)

results, log = [], {}

# -------------------------
# 1. Cognitive/Biological Proxy Dataset
# Example: Daily Minimum Temperatures Melbourne
# -------------------------
print("Loading Daily Minimum Temperatures dataset...")
try:
    url_temp = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv"
    df_temp = pd.read_csv(url_temp)
    sig_temp = df_temp['Temp'].values
    pfd_temp = petrosian_fd(sig_temp)
    hfd_temp = higuchi_fd(sig_temp)
    
    plt.figure(figsize=(12,4))
    plt.plot(sig_temp)
    plt.title(f"Daily Minimum Temperatures (Melbourne) — PFD={pfd_temp:.3f}, HFD={hfd_temp:.3f}")
    plt.tight_layout()
    plt.savefig("plots_hhf/Melbourne_Temperatures.png")
    plt.close()
    
    results.append({"dataset":"Daily_Minimum_Temperatures_Melbourne",
                    "n_points":len(sig_temp),
                    "PFD":pfd_temp,
                    "HFD":hfd_temp})
    log["Daily_Minimum_Temperatures_Melbourne"] = {"n_points":len(sig_temp),"PFD":pfd_temp,"HFD":hfd_temp}
    print(f"Processed Daily_Minimum_Temperatures_Melbourne: PFD={pfd_temp:.3f}, HFD={hfd_temp:.3f}")
except Exception as e:
    print("Error processing temperature dataset:", e)

# -------------------------
# 2. Microtubule Oscillation Simulation
# -------------------------
print("Simulating microtubule oscillation dataset...")
try:
    np.random.seed(42)
    t = np.linspace(0, 100, 2000) # time vector
    # Simulate microtubule oscillation: damped sine wave + noise
    f0, decay = 0.5, 0.01
    sig_mt = np.sin(2*np.pi*f0*t) * np.exp(-decay*t) + 0.05*np.random.randn(len(t))
    
    pfd_mt = petrosian_fd(sig_mt)
    hfd_mt = higuchi_fd(sig_mt)
    
    plt.figure(figsize=(12,4))
    plt.plot(t, sig_mt)
    plt.title(f"Simulated Microtubule Oscillations — PFD={pfd_mt:.3f}, HFD={hfd_mt:.3f}")
    plt.tight_layout()
    plt.savefig("plots_hhf/Microtubule_Oscillations.png")
    plt.close()
    
    results.append({"dataset":"Simulated_Microtubule_Oscillations",
                    "n_points":len(sig_mt),
                    "PFD":pfd_mt,
                    "HFD":hfd_mt})
    log["Simulated_Microtubule_Oscillations"] = {"n_points":len(sig_mt),"PFD":pfd_mt,"HFD":hfd_mt}
    print(f"Processed Simulated_Microtubule_Oscillations: PFD={pfd_mt:.3f}, HFD={hfd_mt:.3f}")
except Exception as e:
    print("Error processing microtubule dataset:", e)

# -------------------------
# 3. Save Logs and Summary
# -------------------------
df_summary = pd.DataFrame(results)
df_summary.to_csv("logs_hhf/fractal_summary.csv", index=False)

with open("logs_hhf/fractal_log.json","w") as f:
    json.dump(log, f, indent=2)

print("\n--- Fractal Analysis Summary ---")
print(df_summary)

print("\n--- Fractal Analysis Log (JSON) ---")
print(json.dumps(log, indent=2))

print("\nHHF Cognitive / Biological Proxy Validation Complete.")
print("All plots saved in plots_hhf/, logs saved in logs_hhf/")
