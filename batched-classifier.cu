#include <iostream>
#include <string>
#include <cstdlib>
#include "dnn.hpp"

#include "cuda.h"
// #include <cuda_runtime.h
#include "helper_cuda.h"
#include "cuda_runtime.h"
#include "cuda_device_runtime_api.h"
#include "device_launch_parameters.h"

using namespace std;

#ifndef Nb
  #define Nb 10  // Number of batches
#endif

//Define the parameters if not defined externally
#ifndef Nn
  #define Nn 128  // Number of Output Layers
  #define Ni 224  // Number of Input  Layers
#endif

#ifndef Tii
  // Tiling Sizes
  #define Tnn 32  
  #define Tii 32
  //#define Tn 5
  //#define Ti 25
  #define Tn 16
  #define Ti 16
#endif

// #define NUM_THREADS Tii
// #define NUM_BLOCKS Tn

#define NUM_THREADS 32
// #define NUM_BLOCKS 256
#define NUM_BLOCKS (Nn/NUM_THREADS)

// Macros for accessing 1D arrays in classifier kernel
#define Synapse(b, n, i) synapse[(b)*Nn*Ni + (n)*Ni + (i)]
#define Neuron_i(b, i) neuron_i[(b)*Nn + i]
#define Neuron_n(b, n) neuron_n[(b)*Nn + n]

//Arrays:
VTYPE (*synapse)[Nb][Nn][Ni];

VTYPE (*neuron_i)[Nb][Ni];
VTYPE (*neuron_n)[Nb][Nn];
VTYPE (*neuron_n2)[Nb][Nn];


// VTYPE synapse[Nb][Nn][Ni] __attribute__((aligned(64)));
// VTYPE neuron_i[Nb][Ni] __attribute__((aligned(64)));
// VTYPE neuron_n[Nb][Nn] __attribute__((aligned(64))),    neuron_n2[Nb][Nn] __attribute__((aligned(64)));

void fill_classifier(VTYPE (&synapse)[Nb][Nn][Ni], VTYPE (&neuron_i)[Nb][Ni], 
    VTYPE (&neuron_n)[Nb][Nn],   VTYPE (&neuron_n2)[Nb][Nn]) {
  for(int b = 0; b < Nb; ++b) {
    for(int n = 0; n < Nn; ++n) {
      for(int i = 0; i < Ni; ++i) {
        synapse[b][n][i] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX) - 0.5f;
      }
    }
  }
  for(int b = 0; b < Nb; ++b) {
    for(int i = 0; i < Ni; ++i) {
      neuron_i[b][i] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX) - 0.5f;
    }
  }
  for(int b = 0; b < Nb; ++b) {
    for(int n = 0; n < Nn; ++n) {
      neuron_n[b][n] = 0; //i;
      neuron_n2[b][n] = 0; //i;
    }
  }
}

void classifier_layer(VTYPE (&synapse)[Nn][Ni], VTYPE (&neuron_i)[Ni], VTYPE (&neuron_n)[Nn]) {
  // int total_calc=0;
  for (int n = 0; n < Nn; n++) {
    VTYPE temp=0;
    for (int i = 0; i < Ni; i++) {
      temp += synapse[n][i] * neuron_i[i];
    }
    neuron_n[n] = transfer(temp);
  }
}

__global__
void classifier_layer_blocked(const VTYPE *synapse, const VTYPE *neuron_i, 
                              VTYPE *neuron_n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int yidx = blockIdx.y + blockDim.y + threadIdx.y;
//   printf("YIDX = %d", yidx);
  for (int b = yidx-1; b < yidx; ++b){
      for (int n = idx*(Nn/(NUM_THREADS*NUM_BLOCKS)); n < (idx+1)*(Nn/(NUM_THREADS*NUM_BLOCKS)); ++n) {
        VTYPE temp_0=0;
    
        for (int i = 0; i < Ni; ++i) {
          // for (int ii = 0; ii < Ti; ++ii){
            temp_0 += Synapse(b, n, i) * Neuron_i(b, i);  // neuron_i[i];
          // }
        }
        Neuron_n(b, n) = temp_0 > 0 ? temp_0 : temp_0/4;
        // neuron_n[n] = temp_0 > 0 ? temp_0 : temp_0/4;
      }
  }
}

int main(int argc, char** argv) {

  synapse   = (VTYPE (*)[Nb][Nn][Ni]) aligned_malloc(64,Nb*Nn*Ni*sizeof(VTYPE));
  neuron_i  = (VTYPE (*)[Nb][Ni]) aligned_malloc(64,Nb*Ni*sizeof(VTYPE));
  neuron_n  = (VTYPE (*)[Nb][Nn]) aligned_malloc(64,Nb*Nn*sizeof(VTYPE));
  neuron_n2 = (VTYPE (*)[Nb][Nn]) aligned_malloc(64,Nb*Nn*sizeof(VTYPE));


  // Error code to check return values for CUDA calls
  cudaError_t err = cudaSuccess;


  // Initialize arrays for run
  cout << "initializing arrays\n";
  fill_classifier(*synapse,*neuron_i,*neuron_n,*neuron_n2);


  // Allocate and copy to Device arrays
  float* d_synapse = NULL;
  err = cudaMalloc((void**)&d_synapse, Nb*Nn*Ni*sizeof(VTYPE));
  if (err != cudaSuccess) {
    cerr << "failed in allocating device synapse" << endl;
    exit(1);
  }
  err = cudaMemcpy(d_synapse, synapse, Nb*Nn*Ni*sizeof(VTYPE), cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    cerr << "failed in copying device synapse" << endl;
    exit(1);
  }

  float* d_neuron_i = NULL;
  err = cudaMalloc((void**)&d_neuron_i, Nb*Ni*sizeof(VTYPE));
  if (err != cudaSuccess) {
    cerr << "failed in allocating device neuron_i" << endl;
    exit(1);
  }
  err = cudaMemcpy(d_neuron_i, neuron_i, Nb*Ni*sizeof(VTYPE), cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    cerr << "failed in copying device neuron_i" << endl;
    exit(1);
  }

  float* d_neuron_n = NULL;
  err = cudaMalloc((void**)&d_neuron_n, Nb*Nn*sizeof(VTYPE));
  if (err != cudaSuccess) {
    cerr << "failed in allocating device neuron_n" << endl;
    exit(1);
  }
  cout << "starting computation\n";

  // Perform and time simple run
//   begin_roi();
//   for (int i = 0; i < Nb; i++) {
//     classifier_layer(((*synapse)[i]),((*neuron_i)[i]),((*neuron_n)[i]));
//   }
//   end_roi(Classifier, 0);

  cout << "simple version complete!\n";

  // Create Stream Objects for concurrent execution
  int nstreams = Nb;
  // allocate and initialize an array of stream handles
  cudaStream_t *streams = (cudaStream_t *) malloc(nstreams * sizeof(cudaStream_t));
  for (int i = 0; i < nstreams; i++)
  {
    checkCudaErrors(cudaStreamCreate(&(streams[i])));
  }

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

  // Perform and time the blocked, distributed run
  dim3 dimGrid(NUM_BLOCKS, Nb, 1);
  dim3 dimThread(NUM_THREADS, 1, 1);

  begin_roi();
  // classifier_layer_blocked(synapse,neuron_i,neuron_n2);
  classifier_layer_blocked<<<dimGrid, dimThread>>>(d_synapse, d_neuron_i, d_neuron_n);

//   if (!CONCURRENT) {
//     for (int i = 0; i < Nb; ++i) {
//       classifier_layer_blocked<<<dimGrid, dimThread>>>(&(d_synapse[order[i]*Nn*Ni]), 
//                                                         &(d_neuron_i[order[i]*Ni]), 
//                                                         &(d_neuron_n[order[i]*Nn]));
//       cudaDeviceSynchronize();
//     }
//     cout << "seq\n";
//   }
//   else {
//     for (int i = 0; i < Nb; i++) {
//       classifier_layer_blocked<<<dimGrid, dimThread, 0, streams[i]>>>(&(d_synapse[order[i]*Nn*Ni]), 
//                                                         &(d_neuron_i[order[i]*Ni]), 
//                                                         &(d_neuron_n[order[i]*Nn]));
//     }
//     cout << "conc\n";
//   }

  cudaDeviceSynchronize();
  end_roi(Classifier, 1);
  
  err = cudaGetLastError();
  if (err != cudaSuccess) {
    cout << "Failed to launch classifier_layer_blocked kernel" << endl;
    exit(1);
  }
  cout << "blocked computation complete!\n"; 
  
  err = cudaMemcpy(neuron_n2, d_neuron_n, Nb*Nn*sizeof(VTYPE), cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) {
    cout << "Failed to copy d_neuron_n from device to host" << endl;
    cout << cudaGetErrorString(err) << endl;
    exit(1);
  }

  // Compare results
//   compare((VTYPE*)*neuron_n,(VTYPE*)*neuron_n2, Nb*Nn, Classifier, 1);
  compare2(Classifier, 1);
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

  cout << "Done!" << endl;
  return 0;
}

