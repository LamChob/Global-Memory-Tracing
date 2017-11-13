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

def create_bins(tmap):
    histogram =[]
    for sk in tmap.values():
        for cta in sk:
            for rk in sk[cta]:
                for rcta in sk[cta][rk].values():
                    recv = rcta 
                    histogram.append(recv["size"])
    return histogram

type_enut = {
   0 : "Load",
   1 : "Store"
}
pp = pprint.PrettyPrinter(indent=2)
tmap = pickle.load( open(sys.argv[1], "rb"))
#print(len(tmap))
hist = create_bins(tmap)
uniques = len(set(hist))

if uniques > 100:
    bincnt = uniques
else:
    bincnt = 'auto'
plt.style.use('ggplot')
plt.hist(hist, bins=100, facecolor='g', alpha=0.75, rwidth=0.8)
plt.yscale('log')
plt.xlabel('bytes transferred')
plt.ylabel('Occurences')
plt.title('Transfer Size Histogram')
plt.show()
