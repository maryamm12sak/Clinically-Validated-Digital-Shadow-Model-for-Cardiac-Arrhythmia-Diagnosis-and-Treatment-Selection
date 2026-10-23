import wfdb
import numpy as np

rec = wfdb.rdrecord('00001_hr', pn_dir='ptb-xl/1.0.3/records500/00000')
np.savetxt("ecg_00001_500hz.csv", rec.p_signal, delimiter=",",
           header=",".join(rec.sig_name), comments="")
print("Saved ecg_00001.csv")