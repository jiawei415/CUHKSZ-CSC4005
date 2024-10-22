//
// Created by Yang Yufan on 2023/10/07.
// Email: yufanyang1@link.cuhk.edu.cn
//
// MPI + OpenMp + SIMD + Reordering Matrix Multiplication
//

#include <mpi.h>  // MPI Header
#include <omp.h> 
#include <immintrin.h>
#include <stdexcept>
#include <chrono>
#include "matrix.hpp"

#define MASTER 0

#include <iostream>
#include <cstring>

Matrix matrix_multiply_mpi(const Matrix& matrix1, const Matrix& matrix2, int numtasks, int taskid) {
    if (matrix1.getCols() != matrix2.getRows()) {
        throw std::invalid_argument(
            "Matrix dimensions are not compatible for multiplication.");
    }

    size_t M = matrix1.getRows(), K = matrix1.getCols(), N = matrix2.getCols();

    Matrix result(M, N);

    int tile_size = M / numtasks;
    int tile_start = taskid * tile_size;
    int tile_end = (taskid == numtasks - 1) ? M : (taskid + 1) * tile_size;

    // Perform tiled matrix multiplication
    for (int i = tile_start; i < tile_end; i++) {
        for (int j = 0; j < N; j++) {
            __m256d sum = _mm256_setzero_pd();
            for (int k = 0; k < K; k += 4) {
                __m256d a = _mm256_loadu_pd(&matrix1(i, k));
                __m256d b = _mm256_broadcast_sd(&matrix2(k, j));
                sum = _mm256_add_pd(sum, _mm256_mul_pd(a, b));
            }
            double temp[4];
            _mm256_storeu_pd(temp, sum);
            for (int k = 0; k < 4; k++) {
                result(i, j) += temp[k];
            }
        }
    }

    // // Synchronize the results
    // if (numtasks > 1) {
    //     double* sendbuf = result.getData();
    //     double* recvbuf = new double[M * N];
    //     std::memcpy(recvbuf, sendbuf, M * N * sizeof(double));
    //     MPI_Allreduce(sendbuf, recvbuf, M * N, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    //     std::memcpy(sendbuf, recvbuf, M * N * sizeof(double));
    //     delete[] recvbuf;
    // }

    return result;
}

int main(int argc, char** argv) {
    // Verify input argument format
    if (argc != 5) {
        throw std::invalid_argument(
            "Invalid argument, should be: ./executable thread_num "
            "/path/to/matrix1 /path/to/matrix2 /path/to/multiply_result\n");
    }

    // Start the MPI
    MPI_Init(&argc, &argv);
    // How many processes are running
    int numtasks;
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
    // What's my rank?
    int taskid;
    MPI_Comm_rank(MPI_COMM_WORLD, &taskid);
    // Which node am I running on?
    int len;
    char hostname[MPI_MAX_PROCESSOR_NAME];
    MPI_Get_processor_name(hostname, &len);
    MPI_Status status;

    int thread_num = atoi(argv[1]);
    omp_set_num_threads(thread_num);

    // Read Matrix
    const std::string matrix1_path = argv[2];

    const std::string matrix2_path = argv[3];

    const std::string result_path = argv[4];

    Matrix matrix1 = Matrix::loadFromFile(matrix1_path);

    Matrix matrix2 = Matrix::loadFromFile(matrix2_path);

    auto start_time = std::chrono::high_resolution_clock::now();
    if (taskid == MASTER) {
        Matrix result = matrix_multiply_mpi(matrix1, matrix2, numtasks, taskid);

        // Your Code Here for Synchronization!

        for (int i = 1; i < numtasks; i++) {
            Matrix temp(M, N);
            MPI_Recv(temp.getData(), M * N, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, &status);
            result += temp;
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        auto elapsed_time =
            std::chrono::duration_cast<std::chrono::milliseconds>(end_time -
                                                                  start_time);

        result.saveToFile(result_path);

        std::cout << "Output file to: " << result_path << std::endl;

        std::cout << "Multiplication Complete!" << std::endl;
        std::cout << "Execution Time: " << elapsed_time.count()
                  << " milliseconds" << std::endl;
    } else {
        Matrix result = matrix_multiply_mpi(matrix1, matrix2, numtasks, taskid);

        // Your Code Here for Synchronization!

        MPI_Send(result.getData(), M * N, MPI_DOUBLE, MASTER, 0, MPI_COMM_WORLD);

    }

    MPI_Finalize();
    return 0;
}