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
const unsigned int N = 16; // This code is designed so N can be of the following values: {16, 32, 64, 128, 256, 512, 1024, 2048}
const unsigned int BLOCK_SIZE = 128;  
const float PRECISION = 0.005;

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
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    extern __shared__ float data[];
    
    for (int m = 0; m < pattern_size; m++) {
        data[threadIdx.x * pattern_size + m] = vector[i * pattern_size + m];
    }
    
    __syncthreads();
    
    for(int step = 2; step < pattern_size; step *= 2) {
        for(int j = 0; j < pattern_size; j += step) {
            int first_index = threadIdx.x * pattern_size + j;
            int second_index = first_index + step / 2;
            float first_result = (data[first_index] + data[second_index]) / sqrtf(2);
            float second_result = (data[first_index] - data[second_index]) / sqrtf(2);
            data[first_index] = first_result;
            data[second_index] = second_result;
        }
    }

    __syncthreads();

    for (int m = 0; m < pattern_size; m++) {
        vector[i * pattern_size + m] = data[threadIdx.x * pattern_size + m];
    }
}

void compute_gpu(vector<float>& vector, unsigned int pattern_size, microseconds* const computation_duration) {
    float* vector_gpu;
    unsigned int vector_size_in_bytes = vector.size() * sizeof(float);
    cudaMalloc(&vector_gpu, vector_size_in_bytes);
    cudaMemcpy(vector_gpu, vector.data(), vector_size_in_bytes, cudaMemcpyHostToDevice); 

    dim3 dimBlock(BLOCK_SIZE);
    dim3 dimGrid((SIZE / N + dimBlock.x - 1) / dimBlock.x);
    
    auto start = steady_clock::now();

    computation_kernel<<<dimGrid, dimBlock, BLOCK_SIZE * pattern_size * sizeof(float)>>>(vector_gpu, pattern_size);

    cudaDeviceSynchronize();
    auto end = steady_clock::now();
    
    if (computation_duration != nullptr) {
        *computation_duration = duration_cast<microseconds>(end - start);
    }

    cudaMemcpy(vector.data(), vector_gpu, vector_size_in_bytes, cudaMemcpyDeviceToHost); 

    cudaFree(vector_gpu);
}

int main() {
    vector<float> vector_cpu = create_random_vector(SIZE);
    vector<float> vector_gpu(vector_cpu);

    compute_cpu(vector_cpu, N);

    microseconds computation_duration;

    auto start = steady_clock::now();
    compute_gpu(vector_gpu, N, &computation_duration);
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