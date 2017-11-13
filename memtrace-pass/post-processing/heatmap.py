import os
import sys
import struct
import pprint
import matplotlib.pyplot as plt
import pickle
import math
import time
import numpy as np
from TraceInc import AutoDict


def timer():
   now = time.time()
   return now

type_enut = {
   0 : "Load",
   1 : "Store"
}

if (len(sys.argv) < 2):
    print("ERROR: incorrect number of args: heatmap.py <in:transfermap>")
    exit(-1)
tmap = pickle.load( open(sys.argv[1], "rb"))


cmap = plt.get_cmap('brg')
cmap.set_under(color='white')
sSet = set()
rSet = set()
simple = AutoDict()

for sk in tmap.values():
    for sk in tmap:
        for cta in tmap[sk]:
            for sit in tmap[sk][cta]:
                for rk in tmap[sk][cta][sit]:
                    for rcta in (tmap[sk][cta][sit][rk]):
                        for rit in (tmap[sk][cta][sit][rk][rcta]):
                            if not simple[cta][rcta]:
                                simple[cta][rcta] = 0
                            simple[cta][rcta] += tmap[sk][cta][sit][rk][rcta][rit]["cnt"]

pp = pprint.PrettyPrinter(indent=2)


range = len(simple)
lower = 0
upper = range
if range > 200:
    upper = 200

plot_mat = [[0 for x in sorted(simple)] for x in sorted(simple)]

cta_enum = {}
for (i,e) in enumerate(sorted(simple)):
    cta_enum[e] = i

for (inx,cta) in enumerate(sorted(simple)):
    for rcta in sorted(list(simple[cta])):
        plot_mat[cta_enum[rcta]][inx] = simple[cta][rcta] 

plot_mat = plot_mat[lower:upper]
for v in plot_mat:
    v[:] = v[lower:upper]

plt.style.use('ggplot')
plt.rcParams.update({'font.size': 14})
plt.imshow(plot_mat, cmap=cmap, interpolation='nearest', origin='lower', vmin=1)
plt.xlabel("CTAs writing")
plt.ylabel("CTAs reading")
plt.colorbar()
path = 'plots/heatmap/'
file = (sys.argv[1].split('/'))[-1]
file = file.split('.')[0] + '.pdf'
plt.savefig(path+file, papertype='a4', bbox_inches='tight', orientation='landscape')
plt.show()

