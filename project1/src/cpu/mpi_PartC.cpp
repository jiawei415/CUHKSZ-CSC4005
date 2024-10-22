//
// Created by Yang Yufan on 2023/9/16.
// Email: yufanyang1@link.cuhk.edu.cn
//
// MPI implementation of transforming a JPEG image from RGB to gray
//

#include <memory.h>
#include <chrono>
#include <iostream>
#include <vector>
#include <mpi.h> // MPI Header

#include "../utils.hpp"

#define MASTER 0
#define TAG_GATHER 0

#include <cmath>

// Function to perform bilateral filtering on an image
void bilateralFilter(const JpegSOA& input_jpeg, JpegSOA& output_jpeg, int rank, int size)
{
    // Get image dimensions
    int width = input_jpeg.width;
    int height = input_jpeg.height;

    // Calculate the range parameter for bilateral filtering
    double range_sigma = 20.0;

    // Calculate the spatial parameter for bilateral filtering
    double spatial_sigma = 2.0;

    // Calculate the range and spatial kernels
    std::vector<double> range_kernel(256);
    std::vector<double> spatial_kernel(width * height);
    for (int i = 0; i < 256; i++)
    {
        range_kernel[i] = exp(-0.5 * pow(i / range_sigma, 2));
    }
    for (int i = 0; i < width; i++)
    {
        for (int j = 0; j < height; j++)
        {
            int index = i * height + j;
            int x = index / height;
            int y = index % height;
            int dx = x - i;
            int dy = y - j;
            spatial_kernel[index] = exp(-0.5 * (pow(dx, 2) + pow(dy, 2)) / pow(spatial_sigma, 2));
        }
    }

    // Perform bilateral filtering on the image
    for (int i = rank; i < width; i += size)
    {
        for (int j = 0; j < height; j++)
        {
            int index = i * height + j;

            // Calculate the weighted sum of neighboring pixels
            double sum_r = 0.0;
            double sum_g = 0.0;
            double sum_b = 0.0;
            double sum_weight = 0.0;

            for (int k = 0; k < width; k++)
            {
                for (int l = 0; l < height; l++)
                {
                    int neighbor_index = k * height + l;

                    // Calculate the color difference
                    double color_diff_r = input_jpeg.r_values[index] - input_jpeg.r_values[neighbor_index];
                    double color_diff_g = input_jpeg.g_values[index] - input_jpeg.g_values[neighbor_index];
                    double color_diff_b = input_jpeg.b_values[index] - input_jpeg.b_values[neighbor_index];

                    // Calculate the weighted sum
                    double weight = range_kernel[static_cast<int>(color_diff_r)] *
                                    range_kernel[static_cast<int>(color_diff_g)] *
                                    range_kernel[static_cast<int>(color_diff_b)] *
                                    spatial_kernel[neighbor_index];
                    sum_r += input_jpeg.r_values[neighbor_index] * weight;
                    sum_g += input_jpeg.g_values[neighbor_index] * weight;
                    sum_b += input_jpeg.b_values[neighbor_index] * weight;
                    sum_weight += weight;
                }
            }

            // Normalize the pixel value
            output_jpeg.r_values[index] = static_cast<unsigned char>(sum_r / sum_weight);
            output_jpeg.g_values[index] = static_cast<unsigned char>(sum_g / sum_weight);
            output_jpeg.b_values[index] = static_cast<unsigned char>(sum_b / sum_weight);
        }
    }
}

int main(int argc, char** argv)
{
    if (argc != 3)
    {
        std::cerr << "Invalid argument, should be: ./executable "
                     "/path/to/input/jpeg /path/to/output/jpeg\n";
        return -1;
    }
    // Read input JPEG File
    const char* input_filepath = argv[1];
    JpegSOA input_jpeg = read_jpeg_soa(input_filepath);
    if (input_jpeg.r_values == nullptr)
    {
        std::cerr << "Failed to read input JPEG image\n";
        return -1;
    }
    auto start_time = std::chrono::high_resolution_clock::now();
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Perform bilateral filtering
    JpegSOA output_jpeg(input_jpeg.width, input_jpeg.height);
    bilateralFilter(input_jpeg, output_jpeg, rank, size);

    // Gather the filtered image from all processes
    MPI_Gather(output_jpeg.r_values, output_jpeg.width * output_jpeg.height, MPI_UNSIGNED_CHAR,
               input_jpeg.r_values, output_jpeg.width * output_jpeg.height, MPI_UNSIGNED_CHAR,
               MASTER, MPI_COMM_WORLD);
    MPI_Gather(output_jpeg.g_values, output_jpeg.width * output_jpeg.height, MPI_UNSIGNED_CHAR,
               input_jpeg.g_values, output_jpeg.width * output_jpeg.height, MPI_UNSIGNED_CHAR,
               MASTER, MPI_COMM_WORLD);
    MPI_Gather(output_jpeg.b_values, output_jpeg.width * output_jpeg.height, MPI_UNSIGNED_CHAR,
               input_jpeg.b_values, output_jpeg.width * output_jpeg.height, MPI_UNSIGNED_CHAR,
               MASTER, MPI_COMM_WORLD);

    MPI_Finalize();
    // END: ed8c6549bwf9
    auto end_time = std::chrono::high_resolution_clock::now();
    auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time);
    std::cout << "Transformation Complete!" << std::endl;
    std::cout << "Execution Time: " << elapsed_time.count()
              << " milliseconds\n";
    return 0;
}
