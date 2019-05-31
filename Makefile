CPP=g++

CFLAGS=$(OPT) --std=c++11 -O3
MODULE          := conv1 conv2 class1 class2 cpu-conv1 cpu-conv2 cpu-class1 cpu-class2

.PHONY: all clean

all: $(MODULE)

HEADERS=dnn.hpp

# These tiling parameters are 100% arbitrary, and it may be advantageous to tune/remove/completely-change them for GPU
conv1: convolution.cu
	/usr/local/cuda-10.1/bin/nvcc -ccbin g++ -I /home/${USER}/NVIDIA_CUDA-10.1_Samples/common/inc -m64 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_37,code=sm_37 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75 -gencode arch=compute_75,code=compute_75 $^ $(CFLAGS) -o $@ -DNx=256 -DNy=256 -DKx=3  -DKy=3  -DNi=64  -DNn=64  -DTii=4 -DTi=16  -DTnn=4 -DTn=16 -DTx=4 -DTy=4 -DCONCURRENT=0 -DNb=10

conv2: convolution.cu
	/usr/local/cuda-10.1/bin/nvcc -ccbin g++ -I /home/${USER}/NVIDIA_CUDA-10.1_Samples/common/inc -m64 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_37,code=sm_37 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75 -gencode arch=compute_75,code=compute_75 $^ $(CFLAGS) -o $@ -DNx=14 -DNy=14 -DKx=3  -DKy=3  -DNi=512  -DNn=512 -DTii=32 -DTi=16  -DTnn=32 -DTn=16 -DTx=2 -DTy=2

clean:
	@rm -f $(MODULE) 
