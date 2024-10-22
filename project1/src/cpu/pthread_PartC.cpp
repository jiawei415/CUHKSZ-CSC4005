//
// Created by Liu Yuxuan on 2024/9/10
// Email: yuxuanliu1@link.cuhk.edu.cn
//
// Row-wise Pthread parallel implementation of smooth image filtering of JPEG
//

#include <memory.h>
#include <chrono>
#include <iostream>
#include <pthread.h>

#include "../utils.hpp"

#include <cmath>

// Struct to hold the parameters for each thread
struct ThreadParams
{
    const JpegSOA* input_jpeg;
    JpegSOA* output_jpeg;
    int start_row;
    int end_row;
    double sigma_spatial;
    double sigma_range;
};

// Function to apply bilateral filtering to a row of pixels
void* bilateralFilterRow(void* arg)
{
    ThreadParams* params = static_cast<ThreadParams*>(arg);

    for (int row = params->start_row; row < params->end_row; row++)
    {
        for (int col = 0; col < params->input_jpeg->width; col++)
        {
            double sum_r = 0.0;
            double sum_g = 0.0;
            double sum_b = 0.0;
            double sum_weight = 0.0;

            for (int i = -3; i <= 3; i++)
            {
                for (int j = -3; j <= 3; j++)
                {
                    int neighbor_row = row + i;
                    int neighbor_col = col + j;

                    if (neighbor_row >= 0 && neighbor_row < params->input_jpeg->height &&
                        neighbor_col >= 0 && neighbor_col < params->input_jpeg->width)
                    {
                        double spatial_distance = std::sqrt(i * i + j * j);
                        double range_distance = std::abs(params->input_jpeg->r_values[row][col] - params->input_jpeg->r_values[neighbor_row][neighbor_col]);

                        double weight = std::exp(-(spatial_distance * spatial_distance) / (2 * params->sigma_spatial * params->sigma_spatial) -
                                                 (range_distance * range_distance) / (2 * params->sigma_range * params->sigma_range));

                        sum_r += params->input_jpeg->r_values[neighbor_row][neighbor_col] * weight;
                        sum_g += params->input_jpeg->g_values[neighbor_row][neighbor_col] * weight;
                        sum_b += params->input_jpeg->b_values[neighbor_row][neighbor_col] * weight;
                        sum_weight += weight;
                    }
                }
            }

            params->output_jpeg->r_values[row][col] = sum_r / sum_weight;
            params->output_jpeg->g_values[row][col] = sum_g / sum_weight;
            params->output_jpeg->b_values[row][col] = sum_b / sum_weight;
        }
    }

    pthread_exit(nullptr);
}

int main(int argc, char** argv)
{
    if (argc != 4)
    {
        std::cerr << "Invalid argument, should be: ./executable "
                     "/path/to/input/jpeg /path/to/output/jpeg NUM_THREADS\n";
        return -1;
    }
    const char* input_filename = argv[1];
    std::cout << "Input file from: " << input_filename << "\n";
    // Read input JPEG image
    JpegSOA input_jpeg = read_jpeg_soa(input_filename);
    if (input_jpeg.r_values == nullptr)
    {
        std::cerr << "Failed to read input JPEG image\n";
        return -1;
    }
    auto start_time = std::chrono::high_resolution_clock::now();
    const int NUM_THREADS = std::stoi(argv[3]);
    const int rows_per_thread = input_jpeg.height / NUM_THREADS;

    pthread_t threads[NUM_THREADS];
    ThreadParams threadParams[NUM_THREADS];

    for (int i = 0; i < NUM_THREADS; i++)
    {
        threadParams[i].input_jpeg = &input_jpeg;
        threadParams[i].output_jpeg = &output_jpeg;
        threadParams[i].start_row = i * rows_per_thread;
        threadParams[i].end_row = (i == NUM_THREADS - 1) ? input_jpeg.height : (i + 1) * rows_per_thread;
        threadParams[i].sigma_spatial = 1.0;
        threadParams[i].sigma_range = 10.0;

        pthread_create(&threads[i], nullptr, bilateralFilterRow, &threadParams[i]);
    }

    for (int i = 0; i < NUM_THREADS; i++)
    {
        pthread_join(threads[i], nullptr);
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time);
    std::cout << "Transformation Complete!" << std::endl;
    std::cout << "Execution Time: " << elapsed_time.count()
              << " milliseconds\n";
    return 0;
}
