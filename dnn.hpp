#ifndef DNN_H
#define DNN_H

#include <inttypes.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define VTYPE float

#ifndef Kx
  #define Kx 1
  #define Ky 1
  #define Nx 1
  #define Ny 1
  #define Nb 1
  #define NYPAD 1
  #define NXPAD 1
#endif

enum ProbType {Convolution, Classifier};

static __inline__ uint64_t gettime(void) { 
  struct timeval tv; 
  gettimeofday(&tv, NULL); 
  return (((uint64_t)tv.tv_sec) * 1000000 + ((uint64_t)tv.tv_usec)); 
} 

static uint64_t usec;

__attribute__ ((noinline))  void begin_roi() {
  usec=gettime();
}

__attribute__ ((noinline))  void end_roi(ProbType p, int blocked)   {
  usec=(gettime()-usec);
}


// Is this a leaky relu?
VTYPE transfer(VTYPE i) {
  return (i>0) ? i : i/4;
}

void compare2(ProbType p, int blocked){
  std::cout << "elapsed (sec): " << usec/1000000.0 << "\n";

    float gflops = 0;
    if (p == Classifier){
      gflops = float(Nn) * Ni * 2/ (usec * 1000); 
    }
    else if (p == Convolution){
      float nxpad = Nx;
      float nypad = Ny;
      gflops = nxpad * nypad * Nb * Nn * Ni * Ky * Kx * 2/ (usec * 1000);
    }
    std::cout << "GFlops (MAC=2) " << blocked << ": " << gflops << "\n";
    // std::cout << "GFlops (MAC=1) " << blocked << ": " << gflops/2 << "\n";
}

void compare(VTYPE* neuron1, VTYPE* neuron2, int size, ProbType p, int blocked) {
  bool error = false;
  for(int i = 0; i < size; ++i) {
      VTYPE diff = neuron1[i] - neuron2[i];
      if(diff>0.001f || diff <-0.001f) {
      error = true; 
      break;
    }
  }
  if(error) {
    for(int i = 0; i < size; ++i) {
      std::cout << i << " " << neuron1[i] << ":" << neuron2[i];;

      VTYPE diff = neuron1[i] - neuron2[i];
      if(diff>0.001f || diff <-0.001f) {
        std::cout << " \t\tERROR";
      }
      std::cout << "\n";
    }
  } else {
    std::cout << "results match\n";

    std::cout << "elapsed (sec): " << usec/1000000.0 << "\n";

    float gflops = 0;
    if (p == Classifier){
      gflops = float(Nn) * Ni * 2/ (usec * 1000); 
    }
    else if (p == Convolution){
      float nxpad = Nx;
      float nypad = Ny;
      gflops = nxpad * nypad * Nb * Nn * Ni * Ky * Kx * 2/ (usec * 1000);
    }
    std::cout << "GFlops (MAC=2) " << blocked << ": " << gflops << "\n";
    std::cout << "GFlops (MAC=1) " << blocked << ": " << gflops/2 << "\n";

  }
}

void* aligned_malloc(uint64_t align, uint64_t bytes)  {
  size_t mask = (align-1)^((size_t)-1);
  char* ptr = (((char*)malloc(bytes+align)) + align);
  ptr = (char*) (((size_t)ptr) & mask);
  return (void*) ptr;
}

#endif
