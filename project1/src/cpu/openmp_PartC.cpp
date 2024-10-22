//
// Created by Zhong Yebin on 2023/9/16.
// Email: yebinzhong@link.cuhk.edu.cn
//
// Sequential implementation of converting a JPEG from RGB to gray
// (Strcture-of-Array)
//

#include <memory.h>
#include <chrono>
#include <cmath>
#include <iostream>
#include <omp.h>

#include "../utils.hpp"

#include <omp.h>

// Function to perform bilateral filtering
void bilateralFilter(const JpegSOA& input_jpeg, JpegSOA& output_jpeg, int num_threads)
{
    int width = input_jpeg.width;
    int height = input_jpeg.height;
    
    // Allocate memory for output image
    output_jpeg.r_values = new uint8_t[width * height];
    output_jpeg.g_values = new uint8_t[width * height];
    output_jpeg.b_values = new uint8_t[width * height];
    
    // Set number of threads
    omp_set_num_threads(num_threads);
    
    // Perform bilateral filtering in parallel
    #pragma omp parallel for
    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            // Perform bilateral filtering for each pixel
            double r_value = 0.0;
            double g_value = 0.0;
            double b_value = 0.0;
            double sum = 0.0;
            for (int k = 0; k < height; k++)
            {
                for (int l = 0; l < width; l++)
                {
                    // Calculate the spatial difference
                    double spatial_diff = sqrt(pow(i - k, 2) + pow(j - l, 2));
                    
                    // Calculate the range difference
                    double range_diff = sqrt(pow(input_jpeg.r_values[i * width + j] - input_jpeg.r_values[k * width + l], 2) +
                                            pow(input_jpeg.g_values[i * width + j] - input_jpeg.g_values[k * width + l], 2) +
                                            pow(input_jpeg.b_values[i * width + j] - input_jpeg.b_values[k * width + l], 2));
                    
                    // Calculate the weight using the spatial and range differences
                    double weight = exp(-0.5 * pow(spatial_diff / 2.0, 2) - 0.5 * pow(range_diff / 20.0, 2));
                    
                    // Accumulate the color values
                    r_value += input_jpeg.r_values[k * width + l] * weight;
                    g_value += input_jpeg.g_values[k * width + l] * weight;
                    b_value += input_jpeg.b_values[k * width + l] * weight;
                    
                    // Accumulate the weight
                    sum += weight;
                }
            }
        }
    }
}

int main(int argc, char** argv)
{
    if (argc != 4)
    {
        std::cerr << "Invalid argument, should be: ./executable "
                     "/path/to/input/jpeg /path/to/output/jpeg NUM_THREADS\n";
        return -1;
    }
    // Read JPEG File
    const char* input_filename = argv[1];
    std::cout << "Input file from: " << input_filename << "\n";
    JpegSOA input_jpeg = read_jpeg_soa(input_filename);
    if (input_jpeg.r_values == nullptr)
    {
        std::cerr << "Failed to read input JPEG image\n";
        return -1;
    }
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Perform bilateral filtering
    JpegSOA output_jpeg;
    bilateralFilter(input_jpeg, output_jpeg, std::stoi(argv[3]));
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time);
    std::cout << "Transformation Complete!" << std::endl;
    std::cout << "Execution Time: " << elapsed_time.count()
              << " milliseconds\n";
    return 0;
}
