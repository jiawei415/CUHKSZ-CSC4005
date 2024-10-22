//
// Created by Yang Yufan on 2023/10/07.
// Email: yufanyang1@link.cuhk.edu.cn
//
// SIMD + Reordering Matrix Multiplication
//

#include <immintrin.h>
#include <stdexcept>
#include <chrono>
#include "matrix.hpp"

Matrix matrix_multiply_simd(const Matrix& matrix1, const Matrix& matrix2) {
    if (matrix1.getCols() != matrix2.getRows()) {
        throw std::invalid_argument(
            "Matrix dimensions are not compatible for multiplication.");
    }

    size_t M = matrix1.getRows(), K = matrix1.getCols(), N = matrix2.getCols();

    Matrix result(M, N);

    // Your Code Here!
    // Optimizing Matrix Multiplication 
    // In addition to Memory Locality and Cache Missing,
    // Further Applying SIMD

    // Set the tile size
    const size_t tile_size = 32;

    // Iterate over the tiles
    for (int i = 0; i < M; i += tile_size) {
        for (int j = 0; j < N; j += tile_size) {
            for (int k = 0; k < K; k += tile_size) {
                // Perform SIMD matrix multiplication on the current tile
                for (int ii = i; ii < std::min(i + tile_size, M); ii++) {
                    for (int jj = j; jj < std::min(j + tile_size, N); jj += 8) {
                        __m256d sum = _mm256_setzero_pd();
                        for (int kk = k; kk < std::min(k + tile_size, K); kk++) {
                            __m256d a = _mm256_loadu_pd(&matrix1(ii, kk));
                            __m256d b = _mm256_broadcast_sd(&matrix2(kk, jj));
                            sum = _mm256_fmadd_pd(a, b, sum);
                        }
                        _mm256_storeu_pd(&result(ii, jj), sum);
                    }
                }
            }
        }
    }
    return result;
}

int main(int argc, char** argv) {
    // Verify input argument format
    if (argc != 4) {
        throw std::invalid_argument(
            "Invalid argument, should be: ./executable "
            "/path/to/matrix1 /path/to/matrix2 /path/to/multiply_result\n");
    }

    const std::string matrix1_path = argv[1];

    const std::string matrix2_path = argv[2];

    const std::string result_path = argv[3];

    Matrix matrix1 = Matrix::loadFromFile(matrix1_path);

    Matrix matrix2 = Matrix::loadFromFile(matrix2_path);

    auto start_time = std::chrono::high_resolution_clock::now();

    Matrix result = matrix_multiply_simd(matrix1, matrix2);

    auto end_time = std::chrono::high_resolution_clock::now();
    auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time);

    result.saveToFile(result_path);

    std::cout << "Output file to: " << result_path << std::endl;

    std::cout << "Multiplication Complete!" << std::endl;
    std::cout << "Execution Time: " << elapsed_time.count() << " milliseconds"
              << std::endl;

    return 0;
}