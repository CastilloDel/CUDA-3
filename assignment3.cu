/* 
   Daniel del Castillo de la Rosa
   
   High Performance Computing master - 2022/2023

*/

#include <iostream>
#include <random>
#include <chrono>
#include <vector>
#include <cfloat>

using std::chrono::steady_clock;
using std::chrono::microseconds;
using std::chrono::duration_cast;
using std::vector;

const unsigned int SIZE = pow(2, 20); 
const unsigned int N = 2048; // This code is designed so N can be of the following values: {16, 32, 64, 128, 256, 512, 1024, 2048}
const unsigned int BLOCK_SIZE = 128;  
const float PRECISION = 0.005;

__constant__ int INDEXES[N];

// Computes associativity indexes for the shuffle needed to achieve coalescence
// Example for size 16
// 0 8 4 12 2 10 6 14 1 9 5 13 3 11 7 15
vector<int> get_associativity_indexes(int size) {
    vector<int> result(size, 0);
    for (int i = size / 2; i < size; i++) {
        result[i] = 1;
    }
    for (int subsize = 2; subsize < size; subsize *= 2) {
        for (int i = subsize / 2; i < size; i += subsize) {
            for (int j = 0; j < subsize / 2; j++) {
                result[i + j] += N / subsize;
            }
        }
    }
    return result;
}

vector<float> create_random_vector(unsigned int size) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> distribution(-1000, 1000);

    vector<float> vector(size);
    for (int i = 0; i < size; i++) {
        vector[i] = distribution(gen);
    }
    return vector;
}

void compute_cpu(vector<float>& vector, unsigned int pattern_size) {
    for(int i = 0; i < vector.size(); i += pattern_size) {
        for(int step = 1; step < pattern_size; step *= 2) {
            for(int jump = 0; jump < pattern_size; jump += step * 2) {
                int first_index = i + jump;
                int second_index = first_index + step;
                float first_result = (vector[first_index] + vector[second_index]) / sqrt(2);
                float second_result = (vector[first_index] - vector[second_index]) / sqrt(2);
                vector[first_index] = first_result;
                vector[second_index] = second_result;
            }
        }
    }
}

__global__ void computation_kernel(float* const vector, int pattern_size) {
    int pattern_position = blockIdx.x * pattern_size;

    extern __shared__ float data[];
    
    for (int m = threadIdx.x; m < pattern_size; m += blockDim.x) {
        data[INDEXES[m]] = vector[pattern_position + m];
    }
    
    __syncthreads();
    
    for(int step = 1; step < pattern_size; step *= 2) {
        for (int m = threadIdx.x; m < pattern_size / (step * 2); m += blockDim.x) {
            int first_index = m;
            int second_index = first_index + pattern_size / (step * 2);
            float first_result = (data[first_index] + data[second_index]) / sqrtf(2);
            float second_result = (data[first_index] - data[second_index]) / sqrtf(2);
            data[first_index] = first_result;
            data[second_index] = second_result;
        }
        __syncthreads();
    }
    
    
    for (int m = threadIdx.x; m < pattern_size; m += blockDim.x) {
        vector[pattern_position + m] = data[INDEXES[m]];
    }
}

void compute_gpu(vector<float>& v, const vector<int>& indexes, microseconds* const computation_duration) {
    float* v_gpu;
    unsigned int v_size_in_bytes = v.size() * sizeof(float);
    cudaMalloc(&v_gpu, v_size_in_bytes);
    cudaMemcpy(v_gpu, v.data(), v_size_in_bytes, cudaMemcpyHostToDevice); 

    unsigned int indexes_size_in_bytes = indexes.size() * sizeof(int);
    cudaMemcpyToSymbol(INDEXES, indexes.data(), indexes_size_in_bytes);

    int pattern_size = indexes.size();
    int block_size = pattern_size / 2;
    dim3 dimBlock(block_size > BLOCK_SIZE ? BLOCK_SIZE : (block_size > 32 ? block_size : 32));
    dim3 dimGrid(v.size() / pattern_size);
    
    auto start = steady_clock::now();

    computation_kernel<<<dimGrid, dimBlock, pattern_size * sizeof(float)>>>(v_gpu, pattern_size);

    cudaDeviceSynchronize();
    auto end = steady_clock::now();
    
    if (computation_duration != nullptr) {
        *computation_duration = duration_cast<microseconds>(end - start);
    }

    cudaMemcpy(v.data(), v_gpu, v_size_in_bytes, cudaMemcpyDeviceToHost); 

    cudaFree(v_gpu);
}

int main() {
    vector<float> vector_cpu = create_random_vector(SIZE);
    vector<float> vector_gpu(vector_cpu);

    compute_cpu(vector_cpu, N);

    microseconds computation_duration;

    auto start = steady_clock::now();
    compute_gpu(vector_gpu, get_associativity_indexes(N), &computation_duration);
    auto end = steady_clock::now();

    microseconds total_duration = duration_cast<microseconds>(end - start);

    int count = 0;
    for (auto i = 0; i < vector_cpu.size(); i++) {
        if (abs(vector_cpu[i] - vector_gpu[i]) > PRECISION) {
            std::cout << vector_cpu[i] << " " << vector_gpu[i] << "\n";
            count++;
        }
    }

    if (count != 0) {
        std::cout << "There was an error: The results from the CPU and GPU doesn't match\n";
    }

    std::cout << "Total time: " << total_duration.count() << "us\n";
    std::cout << "Computation time in the GPU: " << computation_duration.count() << "us\n";
}