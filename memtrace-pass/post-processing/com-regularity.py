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

def comRegularity(tmap):
    kSet = AutoDict()
    reg = AutoDict()
    recvSet = []
    for sk in tmap:
        for sit in tmap[sk]:
            for cta in tmap[sk][sit]:
                if not kSet[sk][sit][cta]:
                    kSet[sk][sit][cta] = []
                for rk in tmap[sk][sit][cta]:
                   for rcta in sorted(tmap[sk][sit][cta][rk]):
                        recvSet.append(rcta)
                   a = frozenset(recvSet.copy())
                   kSet[sk][sit][cta] = a   
                   recvSet.clear()
    for sk in kSet:
        reg[sk] = []
        for scta in kSet[sk]:
            reg[sk].append(1/len(set(kSet[sk][scta].values())))

    return reg

pp = pprint.PrettyPrinter(indent=2)
tmap = pickle.load( open(sys.argv[1], "rb"))

reg = comRegularity(tmap)

plt.style.use('ggplot')
for c in reg:
    plt.plot(reg[c], alpha=0.75, label=c)

plt.xlabel('CTA')
plt.legend()
plt.ylabel('1/unique sets')
plt.title('Communication Regularity')
path = 'plots/com-regularity/'
file = (sys.argv[1].split('/'))[-1]
file = file.split('.')[0] + '.pdf'
plt.savefig(path+file, papertype='a4', bbox_inches='tight', orientation='landscape')
plt.show()
