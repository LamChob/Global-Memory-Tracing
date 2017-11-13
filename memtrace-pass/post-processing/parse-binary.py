import os
import sys
import struct
import pprint
#import matplotlib.pyplot as plt
import pickle
import math
import time
from TraceInc import AutoDict
pp = pprint.PrettyPrinter(indent=2)
vols = pickle.load( open(sys.argv[1], "rb"))

pp.pprint(vols["CDV"])
pp.pprint(vols["CCV"])
exit()
lsum = 0
ssum = 0

