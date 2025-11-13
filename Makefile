CXX = g++
OLD_CXXFLAGS = -O3 -march=native -fopenmp -DNDEBUG
CXXFLAGS = -O3 -march=native -fopenmp -DNDEBUG -funroll-loops -ffast-math -ftree-vectorize -fno-signed-zeros -fno-trapping-math
LDFLAGS = -fopenmp

all: optimized/gemm_mine optimized/gemm_ultra

optimized/gemm_mine: optimized/cpp/mine.cpp
	mkdir -p optimized
	$(CXX) $(CXXFLAGS) -o optimized/gemm_mine optimized/cpp/mine.cpp $(LDFLAGS)

optimized/gemm_ultra: optimized/cpp/mine_ultra_optimized.cpp
	mkdir -p optimized
	$(CXX) $(CXXFLAGS) -o optimized/gemm_ultra optimized/cpp/mine_ultra_optimized.cpp $(LDFLAGS)

all_other: optimized/gemm_opt

optimized/gemm_opt: optimized/cpp/gemm_opt.cpp
	mkdir -p optimized
	$(CXX) $(CXXFLAGS) -o optimized/gemm_opt optimized/cpp/gemm_opt.cpp $(LDFLAGS)

clean:
	rm -f optimized/gemm_mine optimized/gemm_ultra

clean_other:
	rm -f optimized/gemm_opt 
