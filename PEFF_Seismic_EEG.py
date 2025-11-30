# Automated PEFF + EEG Validation (Essential Version)
# ==================================================
!pip install numpy pandas matplotlib mne requests

import os, json, requests, numpy as np, pandas as pd, matplotlib.pyplot as plt
import mne

# -------------------------
# Petrosian Fractal Dimension
# -------------------------
def petrosian_fd(sig):
    sig = np.array(sig, dtype=float)
    N = len(sig)
    if N < 3:
        return np.nan
    diff = np.diff(sig)
    sign_changes = np.sum(diff[:-1]*diff[1:] < 0)
    if sign_changes <= 0:
        return np.nan
    return np.log10(N)/(np.log10(N) + np.log10(N/(N + 0.4*sign_changes)))

# -------------------------
# Directories for output
# -------------------------
os.makedirs("plots_peff", exist_ok=True)
os.makedirs("logs_peff", exist_ok=True)

# -------------------------
# 1. Seismic Data (USGS)
# -------------------------
print("Downloading USGS earthquake data...")
usgs_url = "https://earthquake.usgs.gov/fdsnws/event/1/query.csv?starttime=2025-01-01&endtime=2025-01-31&minmagnitude=1"
try:
    df_seis = pd.read_csv(usgs_url)
    mag_series = df_seis['mag'].fillna(0).values
    fd_seis = petrosian_fd(mag_series)

    # Plot
    plt.figure(figsize=(10,4))
    plt.plot(mag_series)
    plt.title(f"Seismic magnitudes — PFD {fd_seis:.3f}")
    plt.xlabel("Event index")
    plt.ylabel("Magnitude")
    plt.tight_layout()
    plt.savefig("plots_peff/seismic_fd.png")
    plt.close()

    # Log
    log_seis = {"n_points": len(mag_series), "PFD": fd_seis}
    print("Seismic FD:", fd_seis)
except Exception as e:
    log_seis = {"error": str(e)}
    print("Error processing seismic data:", e)

# -------------------------
# 2. EEG Data (PhysioNet Example)
# -------------------------
print("Downloading EEG sample from PhysioNet...")
edf_file = "S001R01.edf"
edf_url = "https://physionet.org/files/eegmmidb/1.0.0/S001/S001R01.edf?download"
try:
    r = requests.get(edf_url)
    with open(edf_file, "wb") as f: f.write(r.content)

    # Load EEG
    raw = mne.io.read_raw_edf(edf_file, preload=True, verbose=False)
    raw.pick_types(eeg=True)
    raw.resample(100)  # downsample for speed

    fd_eeg = {}
    for ch in raw.ch_names:
        sig = raw.get_data(picks=[ch])[0]
        fd_val = petrosian_fd(sig)
        fd_eeg[ch] = fd_val

        # Optional: plot first few channels
        if raw.ch_names.index(ch) < 3:
            plt.figure(figsize=(10,3))
            plt.plot(sig)
            plt.title(f"{ch} — PFD {fd_val:.3f}")
            plt.tight_layout()
            plt.savefig(f"plots_peff/EEG_{ch}.png")
            plt.close()
    print("EEG PFD computed for", len(fd_eeg), "channels")
except Exception as e:
    fd_eeg = {"error": str(e)}
    print("Error processing EEG data:", e)

# -------------------------
# 3. Save Logs and Summary
# -------------------------
summary = {
    "seismic": log_seis,
    "EEG": fd_eeg
}

with open("logs_peff/peff_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

# Optional CSV summary
df_summary = pd.DataFrame({
    "dataset": ["seismic"] + list(fd_eeg.keys()),
    "PFD": [fd_seis] + [fd_eeg[ch] for ch in fd_eeg.keys()]
})
df_summary.to_csv("logs_peff/peff_summary.csv", index=False)

print("PEFF + EEG essential experiment complete. Logs and plots saved.")
