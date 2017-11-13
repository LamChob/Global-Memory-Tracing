#ifndef TRACE_UTILS_H
#define TRACE_UTILS_H

#include <atomic>
#include <string>
#include <thread>
#include <map>
#include <condition_variable>
#include <mutex>


#include <fstream>
#include <iostream>
#include <iomanip>

#include <cuda_runtime.h>
#include <string.h>
#include <stdio.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <signal.h>

constexpr int WarpSize{32};

struct TraceBuffers {
    uint32_t *DFrontInx, *DBackInx; // device reservation (front) and ack (back) index
    uint32_t *HFrontInx, *HBackInx; // host reservation (front) and ack (back) index
    uint64_t *DTraceBuffer, *HTraceBuffer; // pointer to buffer where trace is stored
};


class TraceUtils {
    public:
        TraceUtils(std::string Name = "MemTrace"); 

        // No Copies or Moves allowed!
        TraceUtils(TraceUtils&)  = delete; 
        TraceUtils(TraceUtils&&) = delete; 
        ~TraceUtils(); 

        void start(std::string,cudaStream_t);
        void stop(cudaStream_t);

        uint64_t* getTraceBuffArg(const cudaStream_t S=NULL) {return StreamMap[S]->DTraceBuffer;};
        uint32_t* getFrontInxArg(const cudaStream_t S=NULL)  {return StreamMap[S]->DFrontInx;};
        uint32_t* getBackInxArg(const cudaStream_t S=NULL)   {return StreamMap[S]->DBackInx;};

        uint32_t getSlotSizeArg() const {return SlotSize;};
        uint32_t getSlotPow2Arg() const {return SlotPow2;};

        void createStream(cudaStream_t);
        void destroyStream(cudaStream_t);

    private: 
        void AllocStreamBuffers(cudaStream_t S);
        void FreeStreamBuffers(cudaStream_t S);
        static void TraceConsumer(TraceUtils&, cudaStream_t S);


        // Stream-Individual Members
        std::map< cudaStream_t, std::atomic<bool> > Busy;
        std::map< cudaStream_t, std::atomic<bool> > BusyAck;
        std::map< cudaStream_t, std::atomic<bool> > Terminate;
        std::map< cudaStream_t, std::thread>        TraceThread;
        std::map< cudaStream_t, std::thread>        GzipThread; // stream and PID
        std::map< cudaStream_t, pid_t>              GzipPid;
        std::map< cudaStream_t, TraceBuffers*>      StreamMap;
        std::map< cudaStream_t, int >               StreamNumber;
        std::map< cudaStream_t, std::string >       PipeName;
        std::map< cudaStream_t, std::string >       GzipName;
        std::map< cudaStream_t, std::string >       kName;

        // The same for all Streams
        uint32_t SlotSize; // number if indexes available per slot
        uint32_t SlotPow2; // number of slota as power of two
        uint32_t SlotNum; // Number of slots
        uint32_t RecordSize; // Size in Bytes of one record line
        uint32_t RecordQWord; // Number of QWords in a record
        uint64_t BufferSize; // total amount of reserved bytes

        std::string TraceName;

        uint32_t StreamCounter;
};

extern TraceUtils *__t;
#endif
