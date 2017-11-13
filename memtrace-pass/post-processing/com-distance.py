import os
import sys
import struct
import pprint
import matplotlib.pyplot as plt
import pickle
import math
import time
import statistics
import numpy as np
from TraceInc import AutoDict


def timer():
   now = time.time()
   return now

def comDistance(tmap):
    xmask = 0xFFFFFF00000000 
    ymask = 0x000000FFFF0000
    zmask = 0x0000000000FFFF
    kSet = AutoDict()
    distances = AutoDict()
    for sk in tmap:
        for cta in tmap[sk]:
            if not cta in kSet[sk]:
                kSet[sk][cta] = []
                kSet[sk][cta] = []
                kSet[sk][cta] = []
            sx = (cta & xmask) >> 32
            sy = (cta & ymask) >> 16
            sz = (cta & zmask)
            for sit in tmap[sk][cta]:
                for rk in tmap[sk][cta][sit]:
                   for rcta in sorted(tmap[sk][cta][sit][rk]):
                        rx = (rcta & xmask) >> 32
                        ry = (rcta & ymask) >> 16
                        rz = (rcta & zmask)
                        xd = abs(rx - sx)
                        yd = abs(ry - sy)
                        zd = abs(rz - sz)
                        # euklid/l2 distance
                        kSet[sk][cta].append(math.sqrt(xd**2 + yd**2 + zd**2)) 
    #pp = pprint.PrettyPrinter(indent=2)
    #pp.pprint(kSet)
    for k in kSet:
        for cta in sorted(kSet[k].keys()):
            if not distances[k]:
                    distances[k] = []
            mean = statistics.mean(kSet[k][cta])
            distances[k].append(mean)
    return distances

pp = pprint.PrettyPrinter(indent=2)
tmap = pickle.load( open(sys.argv[1], "rb"))

connect = comDistance(tmap)

#prox = proximity(tmap)
#print(prox)

#reg = regulariy(tmap)
#print(reg)

plt.style.use('ggplot')
for c in connect:
    plt.plot(connect[c], alpha=0.75, label=c)

plt.xlabel('CTA')
plt.legend()
plt.ylabel('Mean L2 Distance')
plt.title('Communication Distance')
path = 'plots/com-distance/'
file = (sys.argv[1].split('/'))[-1]
file = file.split('.')[0] + '.pdf'
plt.savefig(path+file, papertype='a4', bbox_inches='tight', orientation='landscape')
plt.show()
