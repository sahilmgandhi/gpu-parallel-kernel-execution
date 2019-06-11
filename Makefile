CPP=g++

CFLAGS=$(OPT) --std=c++11 -O3 -w
MODULE          := conv1 conv1c opt-conv1 opt-conv1c class1 class1c opt-class1 opt-class1c

.PHONY: all clean

all: $(MODULE)

HEADERS=dnn.hpp

conv1: convolution.cu
	/usr/local/cuda-10.1/bin/nvcc -ccbin g++ -I /home/${USER}/NVIDIA_CUDA-10.1_Samples/common/inc -m64 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_37,code=sm_37 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75 -gencode arch=compute_75,code=compute_75 $^ $(CFLAGS) -o $@ -DNx=256 -DNy=256 -DKx=3  -DKy=3  -DNi=128  -DNn=128  -DTii=4 -DTi=16  -DTnn=4 -DTn=16 -DTx=4 -DTy=4 -DCONCURRENT=0 -DNb=10

conv1c: convolution.cu
	/usr/local/cuda-10.1/bin/nvcc -ccbin g++ -I /home/${USER}/NVIDIA_CUDA-10.1_Samples/common/inc -m64 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_37,code=sm_37 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75 -gencode arch=compute_75,code=compute_75 $^ $(CFLAGS) -o $@ -DNx=256 -DNy=256 -DKx=3  -DKy=3  -DNi=128  -DNn=128  -DTii=4 -DTi=16  -DTnn=4 -DTn=16 -DTx=4 -DTy=4 -DCONCURRENT=1 -DNb=10

opt-conv1: convolution.cu
	/usr/local/cuda-10.1/bin/nvcc -ccbin g++ -I /home/${USER}/NVIDIA_CUDA-10.1_Samples/common/inc -m64 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_37,code=sm_37 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75 -gencode arch=compute_75,code=compute_75 $^ $(CFLAGS) -o $@ -DNx=${NX_PARAM} -DNy=${NY_PARAM} -DKx=3  -DKy=3  -DNi=${NI_PARAM} -DNn=${NN_PARAM} -DTii=4 -DTi=16 -DTnn=4 -DTn=16 -DTx=4 -DTy=4 -DCONCURRENT=0 -DNb=${NUM_BATCHES}

opt-conv1c: convolution.cu
	/usr/local/cuda-10.1/bin/nvcc -ccbin g++ -I /home/${USER}/NVIDIA_CUDA-10.1_Samples/common/inc -m64 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_37,code=sm_37 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75 -gencode arch=compute_75,code=compute_75 $^ $(CFLAGS) -o $@ -DNx=${NX_PARAM} -DNy=${NY_PARAM} -DKx=3  -DKy=3  -DNi=${NI_PARAM} -DNn=${NN_PARAM} -DTii=4 -DTi=16 -DTnn=4 -DTn=16 -DTx=4 -DTy=4 -DCONCURRENT=1 -DNb=${NUM_BATCHES}

class1: classifier.cu 
	/usr/local/cuda-10.1/bin/nvcc -ccbin g++ -I /home/${USER}/NVIDIA_CUDA-10.1_Samples/common/inc -m64 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_37,code=sm_37 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75 -gencode arch=compute_75,code=compute_75 $^ $(CFLAGS) -o $@ -DNi=1024 -DNn=512  -DTii=8 -DTi=32  -DTnn=16  -DTn=8 -DCONCURRENT=0 -DNb=1

class1c: classifier.cu 
	/usr/local/cuda-10.1/bin/nvcc -ccbin g++ -I /home/${USER}/NVIDIA_CUDA-10.1_Samples/common/inc -m64 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_37,code=sm_37 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75 -gencode arch=compute_75,code=compute_75 $^ $(CFLAGS) -o $@ -DNi=1024 -DNn=512  -DTii=8 -DTi=32  -DTnn=16  -DTn=8 -DCONCURRENT=1 -DNb=1

opt-class1: classifier.cu
	/usr/local/cuda-10.1/bin/nvcc -ccbin g++ -I /home/${USER}/NVIDIA_CUDA-10.1_Samples/common/inc -m64 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_37,code=sm_37 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75 -gencode arch=compute_75,code=compute_75 $^ $(CFLAGS) -o $@ -DNi=${NI_PARAM} -DNn=${NN_PARAM} -DCONCURRENT=0 -DNb=${NUM_BATCHES}

opt-class1c: classifier.cu
	/usr/local/cuda-10.1/bin/nvcc -ccbin g++ -I /home/${USER}/NVIDIA_CUDA-10.1_Samples/common/inc -m64 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_37,code=sm_37 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75 -gencode arch=compute_75,code=compute_75 $^ $(CFLAGS) -o $@ -DNi=${NI_PARAM} -DNn=${NN_PARAM} -DCONCURRENT=1 -DNb=${NUM_BATCHES}

class-batched: batched-classifier.cu
	/usr/local/cuda-10.1/bin/nvcc -ccbin g++ -I /home/${USER}/NVIDIA_CUDA-10.1_Samples/common/inc -m64 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_37,code=sm_37 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75 -gencode arch=compute_75,code=compute_75 $^ $(CFLAGS) -o $@ -DNi=1024 -DNn=512  -DTii=8 -DTi=32  -DTnn=16  -DTn=8 -DNb=10

opt-class-batched: batched-classifier.cu
	/usr/local/cuda-10.1/bin/nvcc -ccbin g++ -I /home/${USER}/NVIDIA_CUDA-10.1_Samples/common/inc -m64 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_37,code=sm_37 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75 -gencode arch=compute_75,code=compute_75 $^ $(CFLAGS) -o $@ -DNi=${NI_PARAM} -DNn=${NN_PARAM} -DCONCURRENT=0 -DNb=${NUM_BATCHES}

conv-batched: batched-convolution.cu
	/usr/local/cuda-10.1/bin/nvcc -ccbin g++ -I /home/${USER}/NVIDIA_CUDA-10.1_Samples/common/inc -m64 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_37,code=sm_37 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75 -gencode arch=compute_75,code=compute_75 $^ $(CFLAGS) -o $@ -DNx=256 -DNy=256 -DKx=3  -DKy=3  -DNi=128  -DNn=128  -DTii=4 -DTi=16  -DTnn=4 -DTn=16 -DTx=4 -DTy=4 -DCONCURRENT=1 -DNb=5

clean:
	@rm -f $(MODULE) 
