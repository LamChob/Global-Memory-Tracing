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

def connectivity(tmap):
    partners = AutoDict()
    recvSet = set()
    sum = 0
    for sk in tmap:
        if not sk in partners:
            partners[sk] = []
        for cta in tmap[sk]:
            for sit in tmap[sk][cta]:
                for rk in tmap[sk][cta][sit]:
                    for rcta in sorted(tmap[sk][cta][sit][rk]):
                        recvSet.add(rcta)
            partners[sk].append(len(recvSet)) 
            recvSet.clear()
            sum = 0

    return partners

pp = pprint.PrettyPrinter(indent=2)
tmap = pickle.load( open(sys.argv[1], "rb"))

connect = connectivity(tmap)

#prox = proximity(tmap)
#print(prox)

#reg = regulariy(tmap)
#print(reg)

plt.style.use('ggplot')
for c in connect:
    plt.plot(connect[c], alpha=0.75, label=c)

plt.xlabel('CTA')
plt.legend()
plt.ylabel('Transaction Members')
plt.title('Communication Degree')
path = 'plots/com-degree/'
file = (sys.argv[1].split('/'))[-1]
file = file.split('.')[0] + '.pdf'
plt.savefig(path+file, papertype='a4', bbox_inches='tight', orientation='landscape')
plt.show()
