//leetgpu code

#include <iostream>
#include <cuda_runtime.h>
using namespace std;

// CUDA kernel for vector addition
global void vectorAdd(const int* A, const int* B, int* C, int n) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < n) {
        C[idx] = A[idx] + B[idx];
    }
}

int main() {
    // 🔹 Hardcoded input vectors
    int h_A[] = {1, 2, 3, 4, 5};
    int h_B[] = {5, 4, 300, 2, 1};
    int size = sizeof(h_A) / sizeof(h_A[0]);

    // 🔹 Host result vector
    int* h_C = (int*)malloc(size * sizeof(int));

    // 🔹 Device memory pointers
    int *d_A, *d_B, *d_C;
    size_t bytes = size * sizeof(int);

    // 🔹 Allocate device memory
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    // 🔹 Copy input data to device
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    // 🔹 Launch kernel
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    vectorAdd<<<blocks, threads>>>(d_A, d_B, d_C, size);

    // 🔹 Copy result back to host
    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);

    // 🔹 Display result
    cout << "Vector A: ";
    for (int i = 0; i < size; i++) cout << h_A[i] << " ";
    cout << "\nVector B: ";
    for (int i = 0; i < size; i++) cout << h_B[i] << " ";
    cout << "\nResult C (A + B): ";
    for (int i = 0; i < size; i++) cout << h_C[i] << " ";
    cout << endl;

    // 🔹 Free memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_C);

    return 0;
}