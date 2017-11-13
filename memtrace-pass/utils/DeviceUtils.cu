#include "TraceUtils.h"
#include "DeviceUtils.h"
#include "cuda.h"

__host__ void trace_start(cudaStream_t stream, cudaError_t status, void *vargs) {
    //TraceUtils* t = (TraceUtils*) vargs->utils;
    __t->start(((const char*)vargs), stream);
}

__host__ void trace_stop(cudaStream_t stream, cudaError_t status, void *vargs) {
    //ksdölTraceUtils* t = (TraceUtils*) vargs;
    __t->stop(stream);
}
