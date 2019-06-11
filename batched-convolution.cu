#include <iostream>
#include <string>
#include <cstdlib>
#include "dnn.hpp"

#include "helper_cuda.h"
#include "cuda_runtime.h"
#include "cuda_device_runtime_api.h"
#include "device_launch_parameters.h"

using namespace std;

//Define the parameters if not defined externally
#ifndef Sy
  #define Sy 1
  #define Sx 1
#endif

#ifndef Tnn
  //Tiling Sizes
  #define Tnn 32
  #define Tn  16
  #define Ti  16

  #define Ty  8
  #define Tx  8
#endif

#ifndef CONCURRENT
  #define CONCURRENT true
#endif

#ifndef Nb
  #define Nb 4
#endif

#define NYPAD (Ny+Ky)
#define NXPAD (Nx+Kx)

#define NYSCL (Ny/Sy)
#define NXSCL (Nx/Sx)


#ifndef NUM_THREADS_Y
  // conv1 is 4, conv2 is 2
  #define NUM_THREADS_Y 1
  #define NUM_BLOCKS_Y 8
  // #define NUM_BLOCKS_Y (Ny/NUM_THREADS_Y)

  // conv1 and conv2 are both 2
  #define NUM_THREADS_X 1
  #define NUM_BLOCKS_X Nb
  // #define NUM_BLOCKS_X (Nx/NUM_THREADS_X)

  // conv1 is 4, conv2 is 8
  #define NUM_THREADS_Z 16

#endif

#define NUM_BLOCKS_Z (Nn/NUM_THREADS_Z)

#define SYNAPSE_SIZE (1L*Nb*Ky*Kx*Nn*Ni)


// #define THREADS_PER_BLOCK (NUM_THREADS_Y*NUM_THREADS_X*NUM_THREADS_Z)

#define Synapse(b, i, j, p, q) (synapse[(b) * Ky*Kx*Ni*Nn + (i)*Kx * Ni * Nn + (j)*Ni*Nn + (p)*Nn + (q)])
#define Neuron_i(b, i, j, p) (neuron_i[(b)*NYPAD*NXPAD*Ni + (i)*NXPAD*Ni + (j)*Ni + (p)])
#define Neuron_n(b, i, j, p) (neuron_n[(b)*NYSCL*NXSCL*Nn + (i)*NXSCL*Nn + (j)*Nn + (p)])

VTYPE (*synapse)[Nb][Ky][Kx][Ni][Nn];

VTYPE (*neuron_i)[Nb][NYPAD][NXPAD][Ni];
VTYPE (*neuron_n)[Nb][NYSCL][NXSCL][Nn];
VTYPE (*neuron_n2)[Nb][NYSCL][NXSCL][Nn];

void fill_convolution_shared_simple(VTYPE (&synapse)[Nb][Ky][Kx][Ni][Nn], 
                                    VTYPE (&neuron_i)[Nb][NYPAD][NXPAD][Ni]) {
  for(int bb = 0; bb < Nb; ++bb) {
    for(int yy = 0; yy < Ky; ++yy) {
      for(int xx = 0; xx < Kx; ++xx) {
        for(int ni = 0; ni < Ni; ++ni) {
          for(int nn = 0; nn < Nn; ++nn) {
            synapse[bb][yy][xx][ni][nn] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX) - 0.5f;
          } } } } }
  for(int bb = 0; bb < Nb; ++bb) {
    for(int yy = 0; yy < NYPAD; ++yy) {
      for(int xx = 0; xx < NXPAD; ++xx) {      
        for(int ni = 0; ni < Ni; ++ni) {
          neuron_i[bb][yy][xx][ni] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX) - 0.5f;
    }  }  }  }
}

__global__
void convolution_layer_blocked(
                              const VTYPE *synapse, 
                              const VTYPE *neuron_i, 
                              VTYPE *neuron_n) {

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int idy = blockIdx.y * blockDim.y + threadIdx.y;
  int idn = blockIdx.z * blockDim.z + threadIdx.z;

  const int ySize = Ny/(NUM_THREADS_Y*NUM_BLOCKS_Y); 
//   const int xSize = Nx/(NUM_THREADS_X*NUM_BLOCKS_X);
  const int nSize = NUM_THREADS_Z;
  
  __shared__ VTYPE sum[NUM_THREADS_Z];
  
  // printf("The idx value is %d", idx);
  for (int b = idx; b < idx+1; ++b){
      for (int y = idy*ySize; y < (idy+1)*ySize; ++y) { // tiling for y;
      
        for (int x = 0; x < Nx; ++x) { // tiling for x;
          int n = idn;
          sum[n % nSize]=0;
          // sliding window;
          for (int ky = 0; ky < Ky; ++ky)
            for (int kx = 0; kx < Kx; ++kx)
              for (int i = 0; i < Ni; ++i) {
                VTYPE sv = Synapse(b, ky, kx, i, n);
                VTYPE nv = Neuron_i(b, ky + y, kx + x, i);
                sum[n % nSize] += sv*nv;
              }
            Neuron_n(b, y, x, n) = sum[n% nSize] > 0 ? sum[n% nSize] : sum[n% nSize]/4;
        }
      }
  }
}

void convolution_layer(VTYPE (&synapse)[Ky][Kx][Ni][Nn], 
                               VTYPE (&neuron_i)[NYPAD][NXPAD][Ni], 
                               VTYPE (&neuron_n)[NYSCL][NXSCL][Nn]) {
  VTYPE sum[Nn]={0};

  // — Original code — (excluding nn, ii loops)
  int yout = 0;
  for (int y = 0; y < Ny; y += Sy) { // tiling for y;
    int xout = 0;
    for (int x = 0; x < Ny; x += Sx) { // tiling for x;
      for (int nn = 0; nn < Nn; nn += Tn) {
        for (int n = nn; n < nn + Tn; n++) {
          sum[n]=0;
        }

        // sliding window;
        for (int ky = 0; ky < Ky; ky++)
          for (int kx = 0; kx < Kx; kx++)
            for (int i = 0; i < Ni; i++)
              for (int n = nn; n < nn + Tn; n++) {
                VTYPE sv = synapse[ky][kx][i][n];
                VTYPE nv = neuron_i[ky + y][kx + x][i];
                sum[n]+=sv*nv;
              }
        for (int n = nn; n < nn + Tn; n++) {
          neuron_n[yout][xout][n] = transfer(sum[n]);
        }
      }
      xout++; 
    }
    yout++;
  }
}

int main(const int argc, const char** argv) {
  cout << "allocating memory\n";

  synapse   = (VTYPE (*)[Nb][Ky][Kx][Ni][Nn])  aligned_malloc(64,  SYNAPSE_SIZE*sizeof(VTYPE));
  neuron_i  = (VTYPE (*)[Nb][NYPAD][NXPAD][Ni])aligned_malloc(64,Nb*NYPAD*NXPAD*Ni*sizeof(VTYPE));
  neuron_n  = (VTYPE (*)[Nb][NYSCL][NXSCL][Nn])aligned_malloc(64,Nb*NYSCL*NXSCL*Nn*sizeof(VTYPE));
  neuron_n2 = (VTYPE (*)[Nb][NYSCL][NXSCL][Nn])aligned_malloc(64,Nb*NYSCL*NXSCL*Nn*sizeof(VTYPE));

  cudaError_t err = cudaSuccess;

  cout << "initializing arrays\n";

  fill_convolution_shared_simple(*synapse,*neuron_i);

  float* d_synapse = NULL;
  err = cudaMalloc((void**)&d_synapse, Nb*Ky*Kx*Nn*Ni*sizeof(VTYPE));
  if (err != cudaSuccess) {
    cerr << "failed in allocating device synapse" << endl;
    exit(1);
  }
  err = cudaMemcpy(d_synapse, synapse, Nb*Ky*Kx*Nn*Ni*sizeof(VTYPE), cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    cerr << "failed in copying device synapse" << endl;
    exit(1);
  }

  float* d_neuron_i = NULL;
  err = cudaMalloc((void**)&d_neuron_i, Nb*NYPAD*NXPAD*Nn*sizeof(VTYPE));
  if (err != cudaSuccess) {
    cerr << "failed in allocating device neuron_i" << endl;
    exit(1);
  }
  err = cudaMemcpy(d_neuron_i, neuron_i, Nb*NYPAD*NXPAD*Nn*sizeof(VTYPE), cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    cerr << "failed in copying device neuron_i" << endl;
    exit(1);
  }

  float* d_neuron_n = NULL;
  err = cudaMalloc((void**)&d_neuron_n, Nb*NYSCL*NXSCL*Nn*sizeof(VTYPE));
  if (err != cudaSuccess) {
    cerr << "failed in allocating device neuron_n" << endl;
    exit(1);
  }

  cout << "starting computation\n";

  // for (int i = 0; i < Nb; ++i) {
  //   convolution_layer(((*synapse)[i]),
  //                     ((*neuron_i)[i]),
  //                     ((*neuron_n)[i]));
  //   cout << "simple: " << i << "\n";
  // }

  int nstreams = Nb;
  // allocate and initialize an array of stream handles
  cudaStream_t *streams = (cudaStream_t *) malloc(nstreams * sizeof(cudaStream_t));
  for (int i = 0; i < nstreams; i++)
  {
    checkCudaErrors(cudaStreamCreate(&(streams[i])));
  }

  //end_roi(Convolution, 0);

  cout << "simple version complete!\n";  

  dim3 dimGrid(NUM_BLOCKS_X, NUM_BLOCKS_Y, NUM_BLOCKS_Z);
  dim3 dimThread(NUM_THREADS_X, NUM_THREADS_Y, NUM_THREADS_Z);

  // randomize the order of the batches
  int order[Nb];
  for (int i = 0; i < Nb; ++i) {
    order[i] = i;
  }
  for (int i=0; i<Nb; i++) {
    int r = rand() % Nb;
    int temp = order[i];
    order[i] = order[r];
    order[r] = temp;
  }
  for (int i = 0; i < Nb; ++i) {
    cout << order[i] << " ";
  }
  cout << "\n";

  //Blocked Version
  begin_roi();
  convolution_layer_blocked<<<dimGrid, dimThread>>>(d_synapse, d_neuron_i, d_neuron_n);
//   if (!CONCURRENT) {
//     for (int i = 0; i < Nb; ++i) {
//       convolution_layer_blocked<<<dimGrid, dimThread>>>(&(d_synapse[order[i]*Ky*Kx*Nn*Ni]), 
//                                                         &(d_neuron_i[order[i]*NYPAD*NXPAD*Nn]), 
//                                                         &(d_neuron_n[order[i]*NYSCL*NXSCL*Nn]));
//       cudaDeviceSynchronize();
//     }
//     cout << "seq\n";
//   }
//   else {
//     for (int i = 0; i < Nb; i++) {
//       convolution_layer_blocked<<<dimGrid, dimThread, 256, streams[i]>>>(&(d_synapse[order[i]*Ky*Kx*Nn*Ni]), 
//                                                         &(d_neuron_i[order[i]*NYPAD*NXPAD*Nn]), 
//                                                         &(d_neuron_n[order[i]*NYSCL*NXSCL*Nn]));
//     }
//     cout << "conc\n";
//   }

  cudaDeviceSynchronize();
  end_roi(Convolution, 1);

  if (err != cudaSuccess) {
    cout << "Failed to launch classifier_layer_blocked kernel" << endl;
    exit(1);
  }

  cout << "here\n";
  
  err = cudaMemcpy(neuron_n2, d_neuron_n, Nb*NYSCL*NXSCL*Nn*sizeof(VTYPE), cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) {
    cout << "Failed to copy d_neuron_n from device to host" << endl;
    cout << cudaGetErrorString(err) << endl;
    exit(1);
  }

  cout << "blocked computation complete!\n";  

  // compare((VTYPE*)*neuron_n,(VTYPE*)*neuron_n2, Nb*NYSCL*NXSCL*Nn, Convolution, 1);
  compare2(Convolution, 1);

  cout << "compare done" << endl;

  free(streams);

  // Free device memory
  err = cudaFree(d_synapse);
  if (err != cudaSuccess) {
    cout << "Failed to free device d_synapse" << endl;
    exit(1);
  }

  err = cudaFree(d_neuron_i);
  if (err != cudaSuccess) {
    cout << "Failed to free device d_neuron_i" << endl;
    exit(1);
  }

  err = cudaFree(d_neuron_n);
  if (err != cudaSuccess) {
    cout << "Failed to free device d_neuron_n" << endl;
    exit(1);
  }

  cout << "done\n";
  return 0;
}

