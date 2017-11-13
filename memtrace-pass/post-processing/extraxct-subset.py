import os
import sys
import struct
import pprint
import pickle
import time
from TraceInc import AutoDict

def timer():
   now = time.time()
   return now

type_enum = {
    0 : "Load",
    1 : "Store",
    2 : "Atomic",
    3 : "AtomicAdd",
    4 : "AtomicSub",
    5 : "AtomicEx",
    6 : "AtomicMin",
    7 : "AtomicMax",
    8 : "AtomicInc",
    9 : "AtomicDec",
    10 : "AtomicCAS",
    11 : "AtomicAnd",
    12 : "AtomicOr",
    13 : "AtomicXor",
    14 : "AtomicUn"
}


# Descriptor
# |32 bit|32 bit        |
# |127 96|95    92|91 64|
# | SMID | Type   |Size |
def parseByteString( record, kName, kernel_completion ):
    rec = {}
    rec["it"] = kernel_completion
    rec["smid"] = hex(struct.unpack("<I", record[4:8])[0])

    typeAndSize = struct.unpack("<I", record[0:4])[0]
    rec["size"] = typeAndSize & 0x0FFFFFFF

    # TODO resolve type to ENUM
    rec["type"] = int(typeAndSize & 0xF0000000)
    rec["type"] >>= 28
    rec["type"] = int(rec["type"])

    # unpack little endian (<) 8 bytes(Q) 
    rec["addr"] = struct.unpack("<Q", record[8:16])[0]
   # rec["cta" ] = hex(struct.unpack("<Q", record[16:24])[0])
    rec["cta" ] = struct.unpack("<Q", record[16:24])[0]
    rec["kernel"] = kName
    return rec
    
def extractSmAddrRanges( addressRange, rec, kName ):
    ## init values for correct compare
    if not addressRange[kName][rec["smid"]][rec["type"]]["max"]:
        addressRange[kName][rec["smid"]][rec["type"]]["max"] = 0
    if not addressRange[kName][rec["smid"]][rec["type"]]["min"]:
        addressRange[kName][rec["smid"]][rec["type"]]["min"] = 1 << 63
    if not addressRange[kName][rec["smid"]][rec["type"]]["cnt"]:    
        addressRange[kName][rec["smid"]][rec["type"]]["cnt"] = 0
    
    addressRange[kName][rec["smid"]][rec["type"]]["cnt"] += 1

    if addressRange[kName][rec["smid"]][rec["type"]]["max"] < rec["addr"] :
        addressRange[kName][rec["smid"]][rec["type"]]["max"] = rec["addr"]
    if addressRange[kName][rec["smid"]][rec["type"]]["min"] > rec["addr"] :
        addressRange[kName][rec["smid"]][rec["type"]]["min"] = rec["addr"]
    return addressRange

# extract subset of reads and writes that can be classified as communication
def comSubset( addressCtaMap ):
    subset = []
    for key in addressCtaMap:
        cta_cnt = 0
        if addressCtaMap[key][-1]["type"] == 1:
            del addressCtaMap[key][-1]
    for key in addressCtaMap:
        cta_cnt = 0
        for rec in addressCtaMap[key]:
            if rec["cta"] == addressCtaMap[key][0]["cta"] and rec["kernel"] == addressCtaMap[key][0]["kernel"]:
                cta_cnt +=1
        if len(addressCtaMap[key]) <= 1 or cta_cnt == len(addressCtaMap[key]):
            subset.append(key)

    for key in subset:
        addressCtaMap.pop(key, None)


#    subset = []
#    for key in addressCtaMap:
#        # get out later reads by same cta
#        delete_index = [] 
#        for inx, rec in enumerate(addressCtaMap[key]):
#            if inx+1 == len(addressCtaMap[key]) and rec["type"] == 1: # mark final write for delete
#                delete_index.append(inx)
#                break
#            if rec["type"] == 1: # delete all reads by same cta in same kernel until value is updated
#                i = inx
#                for r in  addressCtaMap[key][inx+1::]:
#                    i+=1
#                    #if r["type"] > 1: # iterate until next non-read
#                    #    break
#                    if r["cta"] == rec["cta"] and r["kernel"] == rec["kernel"]:
#                       delete_index.append(i) 
#                # end delete lookup
#        for i in sorted(delete_index, reverse=True): # delete all records
#            del addressCtaMap[key][i]
#        if len(addressCtaMap[key]) <= 1:
#            subset.append(key)
        #end record
    # end outer
        
def ctaComVolume( addrMap ): #OUTDATED!
    comVol = AutoDict()
    for key in addrMap:
        for rec in addrMap[key][1::]:
            if not comVol[rec["kernel"]][rec["it"]][rec["cta"]][type_enum[rec["type"]]]:
                comVol[rec["kernel"]][rec["it"]][rec["cta"]][type_enum[rec["type"]]] = 0
            comVol[rec["kernel"]][rec["it"]][rec["cta"]][type_enum[rec["type"]]] += rec["size"]
    return comVol

def ctaDataVolumes( rec, dataVolumes ):#OUTDATED!
    if not dataVolumes[rec["kernel"]][rec["it"]][rec["cta"]][type_enum[rec["type"]]]:
        dataVolumes[rec["kernel"]][rec["it"]][rec["cta"]][type_enum[rec["type"]]] = 0
    dataVolumes[rec["kernel"]][rec["it"]][rec["cta"]][type_enum[rec["type"]]] += rec["size"]

def smidDataVolumes( rec, dataVolumes ):#OUTDATED!
    if not dataVolumes[rec["kernel"]][rec["it"]][rec["smid"]][type_enum[rec["type"]]]:
        dataVolumes[rec["kernel"]][rec["it"]][rec["smid"]][type_enum[rec["type"]]] = 0
    dataVolumes[rec["kernel"]][rec["it"]][rec["smid"]][type_enum[rec["type"]]] += rec["size"]

def kernelDataVolumes( rec, dataVolumes ):#OUTDATED!
    if not dataVolumes[rec["kernel"]][rec["it"]][type_enum[rec["type"]]]:
        dataVolumes[rec["kernel"]][rec["it"]][type_enum[rec["type"]]] = 0
    dataVolumes[rec["kernel"]][rec["it"]][type_enum[rec["type"]]] += rec["size"]

def ctaComVolumes( dataVolumes ):#OUTDATED!
    comVol = AutoDict()
    for addr in dataVolumes:
        for rec in dataVolumes[addr]:
            if not comVol[rec["kernel"]][rec["it"]][rec["cta"]][type_enum[rec["type"]]]:
                comVol[rec["kernel"]][rec["it"]][rec["cta"]][type_enum[rec["type"]]] = 0
            comVol[rec["kernel"]][rec["it"]][rec["cta"]][type_enum[rec["type"]]] += rec["size"]

    return comVol

def smidComVolumes( dataVolumes ):#OUTDATED!
    comVol = AutoDict()
    for addr in dataVolumes:
        for rec in dataVolumes[addr]:
            if not comVol[rec["kernel"]][rec["it"]][rec["smid"]][type_enum[rec["type"]]]:
                comVol[rec["kernel"]][rec["it"]][rec["smid"]][type_enum[rec["type"]]] = 0
            comVol[rec["kernel"]][rec["it"]][rec["smid"]][type_enum[rec["type"]]] += rec["size"]

    return comVol

def kernelComVolumes( dataVolumes ):#OUTDATED!
    comVol = AutoDict()
    for addr in dataVolumes:
        for rec in dataVolumes[addr]:
            if not comVol[rec["kernel"]][rec["it"]][type_enum[rec["type"]]]:
                comVol[rec["kernel"]][rec["it"]][type_enum[rec["type"]]] = 0
            comVol[rec["kernel"]][rec["it"]][type_enum[rec["type"]]] += rec["size"]

    return comVol
    
def addressMap(rec, addrMap):
#    # get a map of all records suspectedly communicating
#    if rec["type"] > 0: # write
#        if rec["addr"] in addrMap:
#            addrMap[rec["addr"]].append(rec)
#        else:
#            addrMap[rec["addr"]] = []
#            addrMap[rec["addr"]].append(rec)
#            return
#    elif rec["type"] == 0:
#        if rec["addr"] in addrMap:
#            addrMap[rec["addr"]].append(rec)
    if rec["type"] == 1: # write
        if rec["addr"] in addrMap:
            last_rec = addrMap[rec["addr"]][-1]
            if rec["it"] > last_rec["it"]: # access
                if rec["cta"] != last_rec["cta"] or rec["kernel"] != last_rec["kernel"]:
                    # someone else accessed later than last rec
                     addrMap[rec["addr"]].append(rec)
                elif rec["cta"] == last_rec["cta"] and rec["kernel"] == last_rec["kernel"]:
                    # selfupdate. Keep latest update
                     addrMap[rec["addr"]][-1] = rec
            elif rec["it"] == last_rec["it"]:
                if last_rec["type"] == 0 and last_rec["cta"] == rec["cta"] and last_rec["kernel"] == rec["kernel"]: # write after read in same bsp step, might be communication after all
                    addrMap[rec["addr"]].append(rec)
            else: 
                if last_rec["type"] == 1 and ( rec["cta"] != last_rec["cta"] or rec["kernel"] != last_rec["kernel"] ):
                    print("Completion boundary violation by WW conflict of: ", rec["kernel"], " ", rec["cta"] , " ", rec["it"], "and", last_rec["kernel"], " ", last_rec["cta"], " ", last_rec["it"])
                    addrMap[rec["addr"]].append(rec)
        else: # first entry, alway keep first writes
            addrMap[rec["addr"]] = []
            addrMap[rec["addr"]].append(rec)
    elif rec["type"] == 0: 
        if rec["addr"] in addrMap:
            # find write, this read corresponds to
            last_write = 0
            for inx,w in enumerate(addrMap[rec["addr"]]):
                if w["type"] > 0:
                    last_write = inx 
            last_rec = addrMap[rec["addr"]][last_write]
            nope = False
            if rec["cta"] != last_rec["cta"] or rec["kernel"] != last_rec["kernel"]: # check correspondace to last write
                for r in addrMap[rec["addr"]][last_write+1::]: # check if this has 
                    if r["cta"] == rec["cta"] and r["it"] == rec["it"]: # check if the is a re-read
                        nope = True 
                if not nope:
                    addrMap[rec["addr"]].append(rec)
            #else:
            #    print("Completion boundary violation by WR in same step: ", rec["kernel"] , "and", last_rec["kernel"])

    else: # keep all atomics
        addrMap[rec["addr"]].append(rec)

def ctaTransferIntersection(addrMap):
    transerDens = AutoDict()
    for key in addrMap:
        for inx, rec in enumerate(addrMap[key]):
            if rec["type"] > 0: # store the following reads
                for r in  addrMap[key][inx+1::]:
                    if r["type"] > 0: # iterate until next write or atmoic
                        break

                    if not transerDens[rec["kernel"]][rec["cta"]][rec["it"]][r["kernel"]][r["cta"]][r["it"]]["cnt"]:
                        transerDens[rec["kernel"]][rec["cta"]][rec["it"]][r["kernel"]][r["cta"]][r["it"]]["cnt"] = 0
                    if not transerDens[rec["kernel"]][rec["cta"]][rec["it"]][r["kernel"]][r["cta"]][r["it"]]["size"]:
                        transerDens[rec["kernel"]][rec["cta"]][rec["it"]][r["kernel"]][r["cta"]][r["it"]]["size"] = 0

                    transerDens[rec["kernel"]][rec["cta"]][rec["it"]][r["kernel"]][r["cta"]][r["it"]]["cnt"] += 1
                    transerDens[rec["kernel"]][rec["cta"]][rec["it"]][r["kernel"]][r["cta"]][r["it"]]["size"] += r["size"]
                # end store
        # end record
    #end outer
    return transerDens

def addrSet(tmap):
    rangeSet = AutoDict()
    for addr in tmap:
        for rec in tmap[addr]:
            if not rangeSet[rec["kernel"]][rec["cta"]][rec["it"]][rec["type"]]:
                rangeSet[rec["kernel"]][rec["cta"]][rec["it"]][rec["type"]]= []
            rangeSet[rec["kernel"]][rec["cta"]][rec["it"]][rec["type"]].append(rec["addr"])
    return rangeSet 


if (len(sys.argv) < 5):
    print("ERROR: incorrect number of args: extrace-subset.py <in:trace> <out:vols> <out:transfermap> <out:addrmap>")
    exit(-1)
    
                
with open(sys.argv[1], "rb") as trace:

    ctaRange    = AutoDict()
    smRange     = AutoDict()
    writeList   = AutoDict()
    densityMap  = AutoDict()

    ctaDataVols = AutoDict()
    smidDataVols = AutoDict()
    kernelDataVols = AutoDict()

    ctaComVols = AutoDict()
    smidComVols = AutoDict()
    kernelComVols = AutoDict()

    addrMap     = AutoDict()
    comSet      = AutoDict()

    recordSize = trace.read(1)
    recordSize = 24
    trace.read(1) #skip LF

    # read kernel name
    cnt = 0
    kernel_completion = 0 #counter for completion to distinguish iterations of same kernel
    kName=""

    byte = chr(ord(trace.read(1)))
    while byte != "\n":
        kName += str(byte)
        byte = chr(ord(trace.read(1)))

    print(kName)

    record = trace.read(recordSize)
    
    # now read
    start = timer()

    while record:
        cnt += 1
        if cnt % 1000000 == 0:
            print(cnt)
        if sum(record) != 0: # kernel brake
            rec = parseByteString(record, kName, kernel_completion)
            ctaDataVolumes( rec, ctaDataVols )
            smidDataVolumes( rec, smidDataVols )
            kernelDataVolumes( rec, kernelDataVols )
            addressMap( rec,  addrMap )

        else: 
            byte = trace.read(1)
            if not byte: break # eof
            byte = chr(ord(trace.read(1)))
            kName=""
            while byte != "\n":
                kName += str(byte)
                byte = trace.read(1)
                if not byte: break # eof
                byte = chr(ord(byte))
            print(kName)
            kernel_completion += 1
            if not byte: break # eof
            #  read new Kernel head
        record = trace.read(recordSize)

    # get only the actual communication
    comSubset(addrMap) 

    end = timer()
    totalTime = end - start
    tp = cnt / totalTime
    print("Total Time: ", totalTime, "s, Rec/s: ", tp)
    pp = pprint.PrettyPrinter(indent=2)

    ctaComVols = ctaComVolumes(addrMap)
    smidComVols = smidComVolumes(addrMap)
    kernelComVols = kernelComVolumes(addrMap)

    pickle.dump(addrSet(addrMap), open(sys.argv[4], "wb"))

    pd = AutoDict()
    pd["CDV"] = ctaDataVols
    pd["SDV"] = smidDataVols
    pd["KDV"] = kernelDataVols

    pd["CCV"] = ctaComVols
    pd["SCV"] = smidComVols
    pd["KCV"] = kernelComVols
    pd["it"]  = kernel_completion
    pickle.dump(pd, open(sys.argv[3], "wb"))

    transferMap = ctaTransferIntersection(addrMap) 
    pickle.dump(transferMap, open(sys.argv[2], "wb"))

