#include <iostream>
#include <vector>
#include <numeric>
__device__ void copy(float* to, float* from, float* add) {
    to[threadIdx.x] = from[threadIdx.x] + add[threadIdx.x];
}

__global__ void saxpy(float a, float* x, float* y, int n) {
  //for(int i = 0; i < n; i += blockDim.x ) {
    __shared__ float sx[1024];
    copy(sx, x, y);
    copy(x, sx, sx);
    y[threadIdx.x] = x[0] + sx[12] * a; 
}

int main(int argc, char* argv[]) {
  int kDataLen = atoi(argv[1]);

  float a = 2.0f;

  std::vector<float> host_x(kDataLen, 1.0f);
  std::vector<float> host_y(kDataLen, 0);
  std::vector<uint64_t> h_buff(kDataLen*6,0);


  long h_load, h_store; h_load = h_store = 0;
  long *d_load, *d_store;
  long *inx;
  uint64_t *buff;


  // Copy input data to device.
  float* device_x;
  float* device_y;
  cudaMalloc(&device_x, kDataLen * sizeof(float));
  cudaMalloc(&device_y, kDataLen * sizeof(float));


  // Launch the kernel.
  int nb = kDataLen / 1024;
  saxpy<<<nb, 1024>>>(a, device_x, device_y, kDataLen / 1024);
  int err = cudaGetLastError(); if ( err != 0 ) std::cout << err << std::endl;

  saxpy<<<nb, 1024>>>(a, device_x, device_y, kDataLen / 1024);
  err = cudaGetLastError(); if ( err != 0 ) std::cout << err << std::endl;
  cudaDeviceSynchronize();

  // Prlong the results.
  long sum = std::accumulate(host_y.begin(), host_y.end(), 0);
  std::cout << "Res " << sum << std::endl;

  cudaDeviceReset();
  return 0;
}
