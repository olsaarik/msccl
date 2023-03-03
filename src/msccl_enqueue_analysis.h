#ifndef MSCCL_ENQUEUE_ANALYSIS_H_
#define MSCCL_ENQUEUE_ANALYSIS_H_

#include "nccl.h"

struct MSCCLEnqueueAnalysisResults {
    int numThreadblocks;
    size_t scratchSize;
};

ncclResult_t runMSCCLEnqueueAnalysis(MSCCLEnqueueAnalysisResults* results, int algorithm_index, size_t rank, size_t ranks, size_t count, size_t datatypeSize);

#endif // MSCCL_ENQUEUE_ANALYSIS_H_
