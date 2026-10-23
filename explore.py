import wfdb
import neurokit2 as nk
import matplotlib.pyplot as plt

# Download one ECG from PTB-XL (needs internet)
rec = wfdb.rdrecord('00001_lr', pn_dir='ptb-xl/1.0.3/records100/00000')

print("Shape:", rec.p_signal.shape)   # (1000, 12) = 10 sec x 100 readings/sec, 12 leads
print("Leads:", rec.sig_name)
print("Sampling rate:", rec.fs, "per second")

# Heart rate from lead II
lead2 = rec.p_signal[:, 1]
signals, info = nk.ecg_process(lead2, sampling_rate=rec.fs)
print("Heart rate:", round(signals["ECG_Rate"].mean()), "bpm")

# Draw all 12 leads
wfdb.plot_wfdb(rec, title="PTB-XL record 1")
plt.show()