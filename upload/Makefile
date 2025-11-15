CXX = g++
CXXFLAGS = -O3 -march=native -fopenmp -DNDEBUG -funroll-loops -ffast-math -ftree-vectorize -fno-signed-zeros -fno-trapping-math
LDFLAGS = -fopenmp

all: optimized/gemm_opt

optimized/gemm_opt: optimized/cpp/gemm_opt.cpp
	mkdir -p optimized
	$(CXX) $(CXXFLAGS) -o optimized/gemm_opt optimized/cpp/gemm_opt.cpp $(LDFLAGS)

clean:
	rm -f optimized/gemm_opt