## Description  
   This project explores various optimization strategies for parallel reduction using CUDA.  
   To run:  
     **mkdir build bin data**  
     **g++ scripts/genMat.cpp -o scripts/genMat**  
     **./scripts/genMat**  
     **make**  
     **make run**  
     **make clean**  
   


**Problem &amp; Solution Design**

The problem is to sum 2x2 size matrices using parallel reduction. Each of our matrix shall be represented as 4 floating point numbers. We shall solve the problem using two different storage designs:

1. Row Major: We shall store each matrix consecutively. Below is a pictorial depiction:

1. Column Major: We shall first store first element of each matrix, then second and so on. Below is a pictorial depiction:



In my solution we shall work on the number of matrices rather than on number of elements. Hence we shall depict length in our program as the number of matrices. Thereof each thread will be mapped to a matrix and shall be responsible for reducing its elements. Below is a pictorial depiction of work of one thread:

In row major:

In column major:

Further to get away from edge case if else inside kernel I have made the program such that before passing data to kernel the length is divisible by 1024 which shall be our fixed block dimension. To do this either we pad with zeros or pre reduce some elements before passing. Decision to do what is controlled by a threshold parameter defined in the header file. Since GPU grid dimensions are limited we cannot just dump all the data on GPU. So I have included a volume parameter inside header that controls the volume of the data passed in one go. This parameter has to be set in accordance with the dimensional capabilities of the GPU. Additionally I have done reduction in a iterative manner, i.e. reducing again and again on the previous output until some limit after which the reduction is done by the CPU to get the final output. This limit is set based on the technical specification of the GPU.

**GPU Specifications**

The server GPU has following specifications that are relevant to our experimentation:  

Name: **Tesla K40m**

Compute capability: **3.5**

Architecture: **Kepler**

Number of SMs: **15**

Total global memory: **11.17 G**

Shared memory per block: **48 K**

Cache memory per block: **16 K**

ECC on: **Yes** (Peak bandwidth: 288 G with ECC off)

Number of warp schedulers: **4 ( x2 )**

ALU lanes for single precision FP arithmetic: **192**

Max X dimension of grid: **2^31 -1**

Max X dimension of block: **1024**

Warp size: **32**

Max threads/block: **1024**

Max blocks/SM: **16**

Max warps/SM: **64**

Max threads/SM: **2048**

Max shared memory/SM: **48 K**

Shared memory banks: **32**

Average instructions per warp - **40**

Average instructions per clock  - **2 (issued)**

Average clock latency per warp - 40/2 - **20**

Occupancy requirement - 20 warps / SM - 20/64 - **31%**

**Experiments**

**Version 0: Naive solution**

In this version I have done no optimization. Just plain reduction.

Row Major:

        Cuda time (ms):   94.259002 ( + 355.37236 memcpy )

        Memory bandwidth: 23.4 G

Column Major:

        Cuda time (ms):   56.760094 ( + 697.77817 memcpy )

        Memory bandwidth: 39.3 G

**Version 1: Warp Compression**

Rather than spreading work on all warps, I have restricted work to starting warps so that more and more warps become free as reduction progresses. But we have a series bank conflict here.

Row Major:

        Cuda time (ms):   158.25463 ( + 355.47869 memcpy )

        Memory bandwidth: 14 G

**Version 2: Memory Bank Conflict**

In this version I distributed the work on last warps instead of beginning ones. This will reduce the bank conflicts.

Row Major:

        Cuda time (ms):   70.42156 ( + 354.88986 memcpy )

        Memory bandwidth: 31 G

**Version 3: Reducing first iteration outside loop**

Since reduction is memory intensive, it is better if we do more memory operations as the computation part proceeds.

Row Major:

        Cuda time (ms):   48.755347 ( + 354.14109 memcpy )

        Memory bandwidth: 46 G

**Version 4: Unrolling inner loop**

Finally I have completely unrolled the inner loop. Since our design was to always have 1024 threads in a block, we don&#39;t need templates for this. But this also requires some memory fencing or making the memory volatile

Row Major:

        Cuda time (ms):   38.88473 ( + 354.45972 memcpy )

        Memory bandwidth: 58 G

Column Major:  

        Cuda time (ms):   28.14195 ( + 612.66528 memcpy )
   
        Memory bandwidth: 78.5 G

**References:**

[1] [https://en.wikipedia.org/wiki/CUDA#Version\_features\_and\_specifications](https://en.wikipedia.org/wiki/CUDA#Version_features_and_specifications)

[2] [https://docs.nvidia.com/gameworks/content/developertools/desktop/analysis/report/cudaexperiments/kernellevel/instructionstatistics.htm](https://docs.nvidia.com/gameworks/content/developertools/desktop/analysis/report/cudaexperiments/kernellevel/instructionstatistics.htm)

[3] [https://devtalk.nvidia.com/default/topic/632471/is-syncthreads-required-within-a-warp-/](https://devtalk.nvidia.com/default/topic/632471/is-syncthreads-required-within-a-warp-/)

