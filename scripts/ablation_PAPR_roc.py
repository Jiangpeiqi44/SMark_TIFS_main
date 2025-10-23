from cProfile import label
import numpy as np
import matplotlib.pyplot as plt
from proplot import rc
from sklearn.metrics import roc_curve,auc

# 统一设置字体
rc["font.family"] = "TeX Gyre Schola"

MACOSKO_COLORS = {
    "Amacrine cells": "#A5C93D",
    "Astrocytes": "#8B006B",
    "Bipolar cells": "#2000D7",
    "Cones": "#538CBA",
    "Fibroblasts": "#8B006B",
    "Horizontal cells": "#B33B19",
    "Microglia": "#8B006B",
    "Muller glia": "#8B006B",
    "Pericytes": "#8B006B",
    "Retinal ganglion cells": "#C38A1F",
    "Rods": "#538CBA",
    "Vascular endothelium": "#8B006B",
}
ZEISEL_COLORS = {
    "Astroependymal cells": "#d7abd4",
    "Cerebellum neurons": "#2d74bf",
    "Cholinergic, monoaminergic and peptidergic neurons": "#9e3d1b",
    "Di- and mesencephalon neurons": "#3b1b59",
    "Enteric neurons": "#1b5d2f",
    "Hindbrain neurons": "#51bc4c",
    "Immature neural": "#ffcb9a",
    "Immune cells": "#768281",
    "Neural crest-like glia": "#a0daaa",
    "Oligodendrocytes": "#8c7d2b",
    "Peripheral sensory neurons": "#98cc41",
    "Spinal cord neurons": "#c52d94",
    "Sympathetic neurons": "#11337d",
    "Telencephalon interneurons": "#ff9f2b",
    "Telencephalon projecting neurons": "#fea7c1",
    "Vascular cells": "#3d672d",
}
MOUSE_10X_COLORS = {
    0: "#FFFF00",
    1: "#1CE6FF",
    2: "#FF34FF",
    3: "#FF4A46",
    4: "#008941",
    5: "#006FA6",
    6: "#A30059",
    7: "#FFDBE5",
    8: "#7A4900",
    9: "#0000A6",
    10: "#63FFAC",
    11: "#B79762",
    12: "#004D43",
    13: "#8FB0FF",
    14: "#997D87",
    15: "#5A0007",
    16: "#809693",
    17: "#FEFFE6",
    18: "#1B4400",
    19: "#4FC601",
    20: "#3B5DFF",
    21: "#4A3B53",
    22: "#FF2F80",
    23: "#61615A",
    24: "#BA0900",
    25: "#6B7900",
    26: "#00C2A0",
    27: "#FFAA92",
    28: "#FF90C9",
    29: "#B903AA",
    30: "#D16100",
    31: "#DDEFFF",
    32: "#000035",
    33: "#7B4F4B",
    34: "#A1C299",
    35: "#300018",
    36: "#0AA6D8",
    37: "#013349",
    38: "#00846F",
}
label_simswap = list(map(int,list(np.load("/home/jpq/SepID-AMP/label_true_simswap.npy"))))
pred_simswap = list(np.load("/home/jpq/SepID-AMP/pred_score_simswap.npy"))
label_stargan = list(map(int,list(np.load("/home/jpq/SepID-AMP/label_true_stargan.npy")))) 
pred_stargan = list(np.load("/home/jpq/SepID-AMP/pred_score_stargan.npy"))
label_hfgi = list(map(int,list(np.load("/home/jpq/SepID-AMP/label_true_hfgi.npy")))) 
pred_hfgi = list(np.load("/home/jpq/SepID-AMP/pred_score_hfgi.npy"))
label_diffface = list(map(int,list(np.load("/home/jpq/SepID-AMP/label_true_diffface.npy")))) 
pred_diffface = list(np.load("/home/jpq/SepID-AMP/pred_score_diffface.npy"))

fpr_sim, tpr_sim, thersholds_sim = roc_curve(label_simswap, pred_simswap)
fpr_sta, tpr_sta, thersholds_sta = roc_curve(label_stargan, pred_stargan)
fpr_hfg, tpr_hfg, thersholds_hfg = roc_curve(label_hfgi, pred_hfgi)
fpr_dif, tpr_dif, thersholds_dif = roc_curve(label_diffface, pred_diffface)
 
roc_auc_sim = auc(fpr_sim, tpr_sim)
roc_auc_sta = auc(fpr_sta, tpr_sta)
roc_auc_hfg = auc(fpr_hfg, tpr_hfg)
roc_auc_dif = auc(fpr_dif, tpr_dif)
 
plt.figure(figsize=(4.5, 4.5))
plt.plot(fpr_sim, tpr_sim, 'ro-', label='SimSwap (AUC = {0:.3f})'.format(roc_auc_sim), lw=1.2)
plt.plot(fpr_sta, tpr_sta, 'go-', label='StarGAN (AUC = {0:.3f})'.format(roc_auc_sta), lw=1.2)
plt.plot(fpr_hfg, tpr_hfg, 'bo-', label='HFGI (AUC = {0:.3f})'.format(roc_auc_hfg), lw=1.2)
plt.plot(fpr_dif, tpr_dif, 'yo-', label='DiffFace (AUC = {0:.3f})'.format(roc_auc_dif), lw=1.2)
 
plt.xlim([-0.05, 1.05]) 
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate',fontsize=15)
plt.ylabel('True Positive Rate',fontsize=15)  
plt.legend(loc="lower right",fontsize=14)
plt.savefig("Ablation_PAPR_roc.png")
'''
x = list(np.arange(0, 20,0.5))[:30]
plt.figure(figsize=(10, 4))
plt.plot(x, y_simswap, 'r^-', label='SimSwap', linewidth=0.5)
plt.plot(x, y_stargan, 'gs-', label='StarGAN', linewidth=0.5)
plt.plot(x, y_hfgi, 'bp-', label='HFGI', linewidth=0.5)
plt.plot(x, y_diffface, 'yx-', label='DiffFace', linewidth=0.5)
plt.xlabel('PAPR threshold', fontsize=14)
plt.ylabel('ACC (%)', fontsize=14)
plt.legend()
plt.savefig("Ablation_PAPR_acc.png", bbox_inches='tight', pad_inches=0.1, dpi=200)
'''