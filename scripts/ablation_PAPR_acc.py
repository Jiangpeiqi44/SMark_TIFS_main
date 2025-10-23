import numpy as np
import matplotlib.pyplot as plt
from proplot import rc

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
y_simswap =  list(np.load("/home/jpq/SepID-AMP/acc_simswap.npy")*100)[:30]
y_stargan = list(np.load("/home/jpq/SepID-AMP/acc_stargan.npy")*100)[:30]
y_hfgi = list(np.load("/home/jpq/SepID-AMP/acc_hfgi.npy")*100)[:30]
y_diffface = list(np.load("/home/jpq/SepID-AMP/acc_diffface.npy")*100)[:30]
x = list(np.arange(0, 20,0.5))[:30]
plt.figure(figsize=(4.5, 4.5))
plt.plot(x, y_simswap, 'ro-', label='SimSwap', linewidth=1.2)
plt.plot(x, y_stargan, 'go-', label='StarGAN', linewidth=1.2)
plt.plot(x, y_hfgi, 'bo-', label='HFGI', linewidth=1.2)
plt.plot(x, y_diffface, 'yo-', label='DiffFace', linewidth=1.2)
plt.xlabel('PAPR threshold', fontsize=15)
plt.ylabel('ACC (%)', fontsize=15)
plt.legend(fontsize=13)
plt.savefig("Ablation_PAPR_acc.png")