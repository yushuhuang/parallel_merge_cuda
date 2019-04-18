#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <assert.h>
#include <sys/time.h>

// __global__ void merge(float *data, float *work, int k)
// {
//   int index = blockIdx.x * blockDim.x + threadIdx.x;
    
//   int l = index * k;
//   int m = l + k / 2;
//   int r = l + k;

//   int first = l;
//   int second = m;

//   for (int i = l; i < r; i++)
//   {
//       if (first < m && (second >= r || data[first] <= data[second]))
//       {
//           work[i] = data[first];
//           first += 1;
//       }
//       else
//       {
//           work[i] = data[second];
//           second += 1;
//       }
//   }
// }

// __global__ void parallel_merge(float *data, float *work)
// {
//   uint half = blockDim.x >> 1;
//   uint pos = blockIdx.x * blockDim.x;
//   uint left_array = threadIdx.x < half ? 1 : 0;

//   float cur = data[pos + threadIdx.x];
//   uint i = 0;
//   uint j = half;

//   if (left_array)
//   {
//     while (i < j)
//     {
//       uint mid = i + (j - i) / 2;
//       if (cur <= data[pos + half + mid])
//         j = mid;
//       else
//         i = mid + 1;
//     }
//     work[pos + threadIdx.x + i] = cur;
//   }
//   else
//   {
//     while (i < j)
//     {
//       uint mid = i + (j - i) / 2;
//       if (cur < data[pos + mid])
//         j = mid;
//       else
//         i = mid + 1;
//     }
//     work[pos + threadIdx.x - half + i] = cur;
//   }
// }

__global__ void parallel_merge(float *data, float *work, int stride)
{
  uint index = threadIdx.x * stride;
  uint pos = blockIdx.x * blockDim.x * stride;
  uint half = blockDim.x * stride >> 1;
  uint left_array = index < half ? 1 : 0;

  for (uint s = 0; s < stride; s++)
  {
    float cur = data[pos + index + s];
    uint i = 0;
    uint j = half;
    
    if (left_array)
    {
      while (i < j)
      {
        uint mid = i + (j - i) / 2;
        if (cur <= data[pos + half + mid])
          j = mid;
        else
          i = mid + 1;
      }
      work[pos + index + s + i] = cur;
    }
    else
    {
      while (i < j)
      {
        uint mid = i + (j - i) / 2;
        if (cur < data[pos + mid])
          j = mid;
        else
          i = mid + 1;
      }
      work[pos + index + s - half + i] = cur;
    }
  }
}


int cuda_sort(int number_of_elements, float *a)
{
  cudaError_t cudaStatus;
	
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		return 1;
	}
  
  float *d_a, *d_b;

  cudaMalloc((void **)&d_a, number_of_elements * sizeof(float));
  cudaMalloc((void **)&d_b, number_of_elements * sizeof(float));
  
  cudaMemcpy(d_a, a, number_of_elements * sizeof(float), cudaMemcpyHostToDevice);
  
  int level = 0;
  for (int k = 2; k <= number_of_elements; k = 2 * k)
  {
    int num_merges = number_of_elements / k;
    if (k <= 1024)
    {
      if (level % 2 == 0)
        parallel_merge<<<num_merges, k>>>(d_a, d_b, 1);
      else
        parallel_merge<<<num_merges, k>>>(d_b, d_a, 1);
    }
    else
    {
      if (level % 2 == 0)
        parallel_merge<<<num_merges, 1024>>>(d_a, d_b, k / 1024);
      else
        parallel_merge<<<num_merges, 1024>>>(d_b, d_a, k / 1024);
    }
    cudaDeviceSynchronize();
    level += 1;
  }

  if (level % 2 == 0)
    cudaMemcpy(a, d_a, number_of_elements * sizeof(float), cudaMemcpyDeviceToHost);
  else
    cudaMemcpy(a, d_b, number_of_elements * sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(d_a);
  cudaFree(d_b);
  return 0;
}