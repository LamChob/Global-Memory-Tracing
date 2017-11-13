#include "TraceUtils.h"
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


TraceUtils *__t = new TraceUtils();

TraceUtils::TraceUtils(std::string Name) {
    TraceName = Name;
    
    SlotPow2    = 2;
    SlotSize    = 1 << 18;
    SlotNum     = 1 << SlotPow2; 
    RecordQWord = 3;
    RecordSize  = sizeof(uint64_t);
    BufferSize  = SlotNum * SlotSize * sizeof(uint64_t);
    
    StreamCounter = 1;
}

TraceUtils::~TraceUtils() {
    // deallocate buffers
    for (auto& e: StreamMap) {
        destroyStream(e.first);
        //kill(GzipPid[e.first], SIGTERM); 
        //remove(PipeName[e.first].c_str());
        free(StreamMap[e.first]);
    }
}

void TraceUtils::createStream(cudaStream_t S=NULL) {
    if (StreamNumber[S]) return; // stream already exists
    StreamNumber[S] = StreamCounter++;
    if (StreamMap[S] == nullptr) {
        AllocStreamBuffers(S);
    } 

    Busy[S]       = false;
    BusyAck[S]    = !Busy[S].load();
    Terminate[S]  = false;
    
    // force construction before thread exists
    PipeName[S] = "/tmp/"; 
    PipeName[S].append(TraceName);
    PipeName[S].append("-pipe-");
    PipeName[S].append(std::to_string(StreamNumber[S]));

    GzipName[S] = "/tmp/"; 
    GzipName[S].append(TraceName);
    GzipName[S].append(std::to_string(StreamNumber[S])); 
    GzipName[S].append(".gz");

//    mkfifo(PipeName[S].c_str(), 0666);
    //std::cout << "Pipe Created" << std::endl;
/*
    GzipThread[S] = std::thread([this, S] {
        this->GzipPid[S] = getpid();
        std::string sys = "gzip -1 < "; 
        sys.append(PipeName[S]);
        sys.append(" > ");
        sys.append(GzipName[S]);
        sys.append(" &");
        std::system(sys.c_str());
    });
    
    std::cout << "GZip process started" << std::endl;
*/
    TraceThread[S] = std::thread(TraceConsumer, std::ref(*this), S);
    std::cout << "Consumer process started" << std::endl;

}
void TraceUtils::destroyStream(cudaStream_t S=NULL) {
    Terminate[S] = true;

    std::cout << "Joining... " << std::flush;
    try {
        TraceThread[S].join();
    } catch (const std::system_error& e) {
        std::cout << "Cought Error: " << e.what() << std::endl;
    }
    std::cout << "Done!" << std::endl;
    FreeStreamBuffers(S);
}

void TraceUtils::AllocStreamBuffers(cudaStream_t S) {
    StreamMap[S] = new TraceBuffers;
    cudaHostAlloc(&(StreamMap[S]->HTraceBuffer), BufferSize, cudaHostAllocMapped);
    cudaHostGetDevicePointer(&(StreamMap[S]->DTraceBuffer), StreamMap[S]->HTraceBuffer, 0);

    cudaHostAlloc(&(StreamMap[S]->HFrontInx), SlotNum*sizeof(uint32_t), cudaHostAllocMapped);
    cudaHostGetDevicePointer(&(StreamMap[S]->DFrontInx), StreamMap[S]->HFrontInx, 0);
    memset(StreamMap[S]->HFrontInx, 0, SlotNum*sizeof(uint32_t));

    cudaHostAlloc(&(StreamMap[S]->HBackInx), SlotNum*sizeof(uint32_t), cudaHostAllocMapped);
    cudaHostGetDevicePointer(&(StreamMap[S]->DBackInx), StreamMap[S]->HBackInx, 0);
    memset(StreamMap[S]->HBackInx, 0, SlotNum*sizeof(uint32_t));
}

void TraceUtils::FreeStreamBuffers(cudaStream_t S) {
    // TODO make init somehow better...
    if ( StreamMap[S] == nullptr ) return;
    cudaFree(StreamMap[S]->HFrontInx); 
    cudaFree(StreamMap[S]->HBackInx); 
    cudaFree(StreamMap[S]->HTraceBuffer);
}

void TraceUtils::start(std::string kname, cudaStream_t S=NULL) {
    //std::cout << "Starting Stream..." << std::flush;
    kName[S] = kname;
    if (Busy[S] == true ) {
        std::cout << "Trying to to start already busy trace!"
            << "stream will not be reset, results may be corrupted!" << std::endl;
        return;
    }
    Busy[S] = true;
    while(Busy[S] != BusyAck[S]); // wair for ack
    //std::cout << "Done" << std::endl;
        
}
void TraceUtils::stop(cudaStream_t S=NULL) {
    //std::cout << "Stopping Stream..." << std::flush;
    Busy[S] = false; // signal stop to tracing thread
    while(Busy[S] != BusyAck[S]); // until ack
    //std::cout << "Done" << std::endl;

    // reset all buffers and pointers
    memset(StreamMap[S]->HBackInx, 0, SlotNum*sizeof(uint32_t));
    memset(StreamMap[S]->HFrontInx, 0, SlotNum*sizeof(uint32_t));
    memset(StreamMap[S]->HTraceBuffer, 0, BufferSize);
    atomic_thread_fence(std::memory_order::memory_order_seq_cst);
}

void TraceUtils::TraceConsumer(TraceUtils& Me, cudaStream_t S) {
    const uint32_t BufferThreshhold {WarpSize*Me.RecordQWord-1};
    char recordSeparator[Me.RecordSize*Me.RecordQWord];
    memset(recordSeparator, 0, Me.RecordSize*Me.RecordQWord);
    const char LF = 10;
    const char len = Me.RecordSize * Me.RecordQWord;

    #ifdef PRINT_LITERAL
    FILE* outpipe = fopen(Me.PipeName[S].c_str(), "wb");
    #else 
    FILE* outpipe = fopen(Me.PipeName[S].c_str(), "w");
    #endif

    #ifdef PRINT_LITERAL
    //fprintf(outpipe, "\n%s\n", Me.kName[S].c_str()); 
    #else 
    fwrite(&len, sizeof(char), 1, outpipe);
    #endif
    

    while(1) {
        if (Me.Terminate[S]) break;
        if (Me.Busy[S]) {
            Me.BusyAck[S] = Me.Busy[S].load();
            #ifdef PRINT_LITERAL
            fprintf(outpipe, "\n%s\n", Me.kName[S].c_str()); 
            #else 
            fwrite(&LF, sizeof(char), 1, outpipe);
            fwrite(Me.kName[S].c_str(), Me.kName[S].size(), 1, outpipe);
            fwrite(&LF, sizeof(char), 1, outpipe);
            #endif
            while(Me.Busy[S]) {
                for(int slot = 0; slot < Me.SlotNum; slot++) {
                    volatile unsigned int *i1 = &(Me.StreamMap[S]->HFrontInx[slot]);
                    volatile unsigned int *i2 = &(Me.StreamMap[S]->HBackInx[slot]);
                    if (*i2 >= Me.SlotSize-BufferThreshhold) {
                        //std::cout << "reset: " << slot << std::endl;
                        int offset = slot*Me.SlotSize;
                        unsigned int inx = *i2;
                        #ifdef PRINT_LITERAL
                        for(int i = 0; i < inx; i += Me.RecordQWord) {
                            fprintf(outpipe, "%016lx %016lx %016lx \n",
                                 Me.StreamMap[S]->HTraceBuffer[offset+i],
                                 Me.StreamMap[S]->HTraceBuffer[offset+i+1],
                                 Me.StreamMap[S]->HTraceBuffer[offset+i+2]
                            );
                        }
                        #else 
                        fwrite(&(Me.StreamMap[S]->HTraceBuffer[offset]), Me.RecordSize, inx, outpipe);
                        #endif

                        *i2=0;
                        *i1= 0;
                        // just to be safe...
                        atomic_thread_fence(std::memory_order::memory_order_seq_cst);
                    } 
                }
            }
            //clear remainnig Buffers
            for(int slot = 0; slot < Me.SlotNum; slot++) {
                volatile unsigned int *i2 = &(Me.StreamMap[S]->HBackInx[slot]);
                unsigned int inx = *i2;
                int offset = slot*Me.SlotSize;
                    #ifdef PRINT_LITERAL
                    for(int i = 0; i < inx; i += 3) {
                        fprintf(outpipe, "%016lx %016lx %016lx \n",
                             Me.StreamMap[S]->HTraceBuffer[offset+i],
                             Me.StreamMap[S]->HTraceBuffer[offset+i+1],
                             Me.StreamMap[S]->HTraceBuffer[offset+i+2]
                        );
                    }
                    #else 
                    fwrite(&(Me.StreamMap[S]->HTraceBuffer[offset]), Me.RecordSize, inx, outpipe);
                    #endif
                    *i2=0;
            }
            #ifdef PRINT_LITERAL
            fprintf(outpipe, "0000000000000000000000000000000000000000000000000000000\n");
            #else 
            fwrite(recordSeparator, 1, Me.RecordSize*Me.RecordQWord, outpipe);
            #endif
        }
        Me.BusyAck[S] = Me.Busy[S].load();
    }
    fclose(outpipe);
    return;
}
