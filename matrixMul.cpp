#include <iostream>
#include <cuda_runtime.h>
using namespace std;

// ✅ Set matrix dimensions here
const int M = 4; // Rows in A
const int N = 2; // Columns in A and rows in B
const int P = 2; // Columns in B

// CUDA kernel to compute C = A × B
global void matrixMultiply(int* A, int* B, int* C, int m, int n, int p) {
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int col = threadIdx.x + blockIdx.x * blockDim.x;

    if (row < m && col < p) {
        int sum = 0;
        for (int k = 0; k < n; ++k) {
            sum += A[row * n + k] * B[k * p + col];
        }
        C[row * p + col] = sum;
    }
}

int main() {
    // Memory allocation sizes
    size_t sizeA = M * N * sizeof(int);
    size_t sizeB = N * P * sizeof(int);
    size_t sizeC = M * P * sizeof(int);

    // ✅ Matrix A[M × N]
    int h_A[M * N] = {
        1, 3, 
        5, 6,
        4, 5,
        5, 6
    };

    // ✅ Matrix B[N × P]
    int h_B[N * P] = {
        1, 5,
        4, 6
       
    };

    int h_C[M * P]; // Resultant matrix C[M × P]

    // Allocate device memory
    int *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, sizeA);
    cudaMalloc(&d_B, sizeB);
    cudaMalloc(&d_C, sizeC);

    // Copy input matrices to device
    cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice);

    // Kernel launch configuration
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((P + 15) / 16, (M + 15) / 16);

    // Run kernel
    matrixMultiply<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, M, N, P);
    cudaDeviceSynchronize();

    // Copy result matrix back to host
    cudaMemcpy(h_C, d_C, sizeC, cudaMemcpyDeviceToHost);

    // ✅ Display all matrices
    cout << "\nMatrix A (" << M << "x" << N << "):\n";
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j)
            cout << h_A[i * N + j] << " ";
        cout << endl;
    }

    cout << "\nMatrix B (" << N << "x" << P << "):\n";
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < P; ++j)
            cout << h_B[i * P + j] << " ";
        cout << endl;
    }

    cout << "\nMatrix C = A × B (" << M << "x" << P << "):\n";
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < P; ++j)
            cout << h_C[i * P + j] << " ";
        cout << endl;
    }

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}