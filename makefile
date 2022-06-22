install:
	mkdir -p build
	#compiling utils
	nvcc --device-c -o build/GPU_utils.o src/utils/GPU_utils.cu -Xcompiler -fPIC
	#compiling clustering
	nvcc --device-c -o build/GPU_DBSCAN.o src/algorithms/clustering/GPU_DBSCAN.cu -Xcompiler -fPIC
	nvcc --device-c -o build/EGG_SynC.o src/algorithms/clustering/EGG_SynC.cu -Xcompiler -fPIC
	nvcc --shared -o libclustering.so \
		src/cpp_wrappers/clustering.cpp \
		src/algorithms/clustering/SynC.cpp \
		src/algorithms/clustering/DBSCAN.cpp \
		src/structure/RTree.cpp \
		src/utils/CPU_math.cpp \
		src/utils/CPU_mem_util.cpp \
		build/EGG_SynC.o \
        build/GPU_DBSCAN.o \
        build/GPU_utils.o \
		-Xcompiler -fPIC
	#compiling projected clustering
	nvcc --device-c -o build/GPU_PROCLUS.o src/algorithms/projected_clustering/GPU_PROCLUS.cu -Xcompiler -fPIC
	nvcc --shared -o libprojectedclustering.so \
		src/cpp_wrappers/projected_clustering.cpp \
		src/algorithms/projected_clustering/PROCLUS.cpp \
		src/utils/CPU_math.cpp \
		src/utils/CPU_mem_util.cpp \
		build/GPU_PROCLUS.o \
        build/GPU_utils.o \
		-Xcompiler -fPIC

	python3 setup.py build_ext --inplace
	sphinx-build -b html docs/source docs/build/html/0.0.5/