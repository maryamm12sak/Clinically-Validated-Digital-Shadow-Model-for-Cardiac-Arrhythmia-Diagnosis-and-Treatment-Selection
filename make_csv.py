import wfdb
import numpy as np

rec = wfdb.rdrecord('00001_lr', pn_dir='ptb-xl/1.0.3/records100/00000')
np.savetxt("ecg_00001.csv", rec.p_signal, delimiter=",",
           header=",".join(rec.sig_name), comments="")
print("Saved ecg_00001.csv")