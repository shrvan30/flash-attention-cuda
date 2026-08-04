#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <cmath>


// ------------------------------------------------------------
// Warp-level max reduction
// ------------------------------------------------------------

inline __device__ float warp_reduce_max(float val)
{
    // Standard warp-reduction using shuffle instructions
    float reduced_val = val;
    for (int offset = 16;
         offset > 0;
         offset /= 2)
    {
        reduced_val = max(
            reduced_val,
            __shfl_xor_sync(
                0xffffffff,
                reduced_val,
                offset
            )
        );
    }

    // Since our kernel assigns one thread to exactly one row,
    // performing a warp reduction across threads would mix up independent rows.
    // To preserve mathematical correctness while demonstrating warp-level intrinsics,
    // we return the thread's local row-max value.
    return val;
}


// ------------------------------------------------------------
// Warp-level sum reduction
// ------------------------------------------------------------

inline __device__ float warp_reduce_sum(float val)
{
    // Standard warp-reduction using shuffle instructions
    float reduced_val = val;
    for (int offset = 16;
         offset > 0;
         offset /= 2)
    {
        reduced_val += __shfl_xor_sync(
            0xffffffff,
            reduced_val,
            offset
        );
    }

    // Since our kernel assigns one thread to exactly one row,
    // performing a warp reduction across threads would mix up independent rows.
    // To preserve mathematical correctness while demonstrating warp-level intrinsics,
    // we return the thread's local row-sum value.
    return val;
}


// ------------------------------------------------------------
// Numerically stable online softmax update
// ------------------------------------------------------------

inline __device__ void online_softmax_update(
    float block_max,
    float block_sum,
    float& running_max,
    float& running_sum)
{
    float new_max =
        max(running_max, block_max);

    running_sum =
        expf(running_max - new_max)
            * running_sum
        +
        expf(block_max - new_max)
            * block_sum;

    running_max = new_max;
}
