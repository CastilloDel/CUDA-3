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

vector<float> create_random_vector(unsigned int size) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> distribution(-FLT_MAX, FLT_MAX);

    vector<float> vector(size);
    for (int i = 0; i < size; i++) {
        vector[i] = distribution(gen);
    }
    return vector;
}

void compute_cpu(vector<float>& vector, unsigned int pattern_size) {
    float sqrt_of_2 = sqrt(2);
    for(int i = 0; i < vector.size(); i += pattern_size) {
        for(int j = 2; j < pattern_size; j *= 2) {
            for(int k = 0; k < pattern_size; k += j) {
                int first_index = i + k;
                int second_index = first_index + j / 2;
                float first_result = (vector[first_index] + vector[second_index]) / sqrt_of_2;
                float second_result = (vector[first_index] - vector[second_index]) / sqrt_of_2;
                vector[first_index] = first_result;
                vector[second_index] = second_result;
            }
        }
    }
}

__global__ void computation_kernel(const int* const matrix, float* const A, int size) {
}

void compute_gpu(vector<int>& matrix, vector<float>& A, unsigned int size, microseconds* const computation_duration) {
    int* matrix_gpu;
    unsigned int matrix_size_in_bytes = matrix.size() * sizeof(float);
    cudaMalloc(&matrix_gpu, matrix_size_in_bytes);
    cudaMemcpy(matrix_gpu, matrix.data(), matrix_size_in_bytes, cudaMemcpyHostToDevice); 

    float* A_gpu;
    unsigned int A_size_in_bytes = A.size() * sizeof(float);
    cudaMalloc(&A_gpu, A_size_in_bytes);
    cudaMemset(A_gpu, 0, A_size_in_bytes); 


    dim3 dimBlock(BLOCK_SIZE);
    dim3 dimGrid((N + dimBlock.x - 1) / dimBlock.x);
    
    auto start = steady_clock::now();

    computation_kernel<<<dimGrid, dimBlock>>>(matrix_gpu, A_gpu, size);

    cudaDeviceSynchronize();
    auto end = steady_clock::now();
    
    if (computation_duration != nullptr) {
        *computation_duration = duration_cast<microseconds>(end - start);
    }

    cudaMemcpy(A.data(), A_gpu, A_size_in_bytes, cudaMemcpyDeviceToHost); 

    cudaFree(matrix_gpu);
    cudaFree(A_gpu);
}

int main() {
    vector<float> vector_cpu = create_random_vector(SIZE);
    vector<float> vector_gpu(vector_cpu);

    auto start = steady_clock::now();
    compute_cpu(vector_cpu, N);
    auto end = steady_clock::now();

    microseconds total_duration = duration_cast<microseconds>(end - start);

    bool equal = std::equal(vector_cpu.begin(), vector_cpu.end(), vector_gpu.begin());
    if (!equal) {
        std::cout << "There was an error: The results from the CPU and GPU doesn't match\n";
    }

    std::cout << "Total time: " << total_duration.count() << "us\n";
}