nvcc=/usr/local/cuda-10.1/bin/nvcc
cudalib=/usr/local/cuda-10.1/lib64
tensorflow=/home/cm21/anaconda3/envs/deformnet37/lib/python3.7/site-packages/tensorflow_core/include
TF_LIB=/home/cm21/anaconda3/envs/deformnet37/lib/python3.7/site-packages/tensorflow_core
all: tf_nndistance_so.so tf_nndistance_g.cu.o 

tf_nndistance_so.so: tf_nndistance_g.cu.o tf_nndistance.cpp
	g++ -std=c++11 tf_nndistance.cpp tf_nndistance_g.cu.o -o tf_nndistance_so.so -shared -fPIC -I $(tensorflow) -lcudart -L $(cudalib) -O2 -L $(TF_LIB) -ltensorflow_framework -D_GLIBCXX_USE_CXX11_ABI=0

tf_nndistance_g.cu.o: tf_nndistance_g.cu
	$(nvcc) -D_GLIBCXX_USE_CXX11_ABI=0 -std=c++11 -c -o tf_nndistance_g.cu.o tf_nndistance_g.cu -I $(tensorflow) -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -O2

clean:
	rm tf_nndistance_so.so
	rm  *.cu.o 
