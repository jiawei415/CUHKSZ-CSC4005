//
// Created by Liu Yuxuan on 2024/9/11
// Modified from Zhong Yebin's PartB on 2023/9/16
//
// Email: yebinzhong@link.cuhk.edu.cn
// Email: yuxuanliu1@link.cuhk.edu.cn
//
// CUDA implementation of bilateral filtering on JPEG image
//

#include <iostream>
#include <cmath>
#include <chrono>
#include <cuda_runtime.h>

#include "../utils.hpp"

/**
 * Demo kernel device function to clamp pixel value
 * 
 * You may mimic this to implement your own kernel device functions
 */
__device__ unsigned char d_clamp_pixel_value(float value)
{
    return value > 255 ? 255
           : value < 0 ? 0
                       : static_cast<unsigned char>(value);
}

__global__ void bilateral_filter(const unsigned char* input_r,
                                 const unsigned char* input_g,
                                 const unsigned char* input_b,
                                 unsigned char* output_r,
                                 unsigned char* output_g,
                                 unsigned char* output_b,
                                 int width,
                                 int height,
                                 float sigma_spatial,
                                 float sigma_range)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < width && row < height)
    {
        float sum_r = 0.0f;
        float sum_g = 0.0f;
        float sum_b = 0.0f;
        float weight_sum = 0.0f;

        int center_r = input_r[row * width + col];
        int center_g = input_g[row * width + col];
        int center_b = input_b[row * width + col];

        for (int i = -2; i <= 2; i++)
        {
            for (int j = -2; j <= 2; j++)
            {
                int current_col = col + j;
                int current_row = row + i;

                if (current_col >= 0 && current_col < width && current_row >= 0 && current_row < height)
                {
                    int neighbor_r = input_r[current_row * width + current_col];
                    int neighbor_g = input_g[current_row * width + current_col];
                    int neighbor_b = input_b[current_row * width + current_col];

                    float spatial_distance = sqrtf((i * i) + (j * j));
                    float range_distance = sqrtf(((center_r - neighbor_r) * (center_r - neighbor_r)) +
                                                 ((center_g - neighbor_g) * (center_g - neighbor_g)) +
                                                 ((center_b - neighbor_b) * (center_b - neighbor_b)));

                    float weight = expf(-(spatial_distance * spatial_distance) / (2 * sigma_spatial * sigma_spatial) -
                                        (range_distance * range_distance) / (2 * sigma_range * sigma_range));

                    sum_r += neighbor_r * weight;
                    sum_g += neighbor_g * weight;
                    sum_b += neighbor_b * weight;
                    weight_sum += weight;
                }
            }
        }

        output_r[row * width + col] = d_clamp_pixel_value(sum_r / weight_sum);
        output_g[row * width + col] = d_clamp_pixel_value(sum_g / weight_sum);
        output_b[row * width + col] = d_clamp_pixel_value(sum_b / weight_sum);
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
    // Read input JPEG image in structure-of-array form
    const char* input_filename = argv[1];
    std::cout << "Input file from: " << input_filename << "\n";
    JpegSOA input_jpeg = read_jpeg_soa(input_filename);
    if (input_jpeg.r_values == nullptr)
    {
        std::cerr << "Failed to read input JPEG image\n";
        return -1;
    }

    // Set up CUDA device
    int device = 0;
    cudaSetDevice(device);

    // Allocate memory on the device
    unsigned char* d_input_r;
    unsigned char* d_input_g;
    unsigned char* d_input_b;
    unsigned char* d_output_r;
    unsigned char* d_output_g;
    unsigned char* d_output_b;
    cudaMalloc((void**)&d_input_r, input_jpeg.width * input_jpeg.height * sizeof(unsigned char));
    cudaMalloc((void**)&d_input_g, input_jpeg.width * input_jpeg.height * sizeof(unsigned char));
    cudaMalloc((void**)&d_input_b, input_jpeg.width * input_jpeg.height * sizeof(unsigned char));
    cudaMalloc((void**)&d_output_r, input_jpeg.width * input_jpeg.height * sizeof(unsigned char));
    cudaMalloc((void**)&d_output_g, input_jpeg.width * input_jpeg.height * sizeof(unsigned char));
    cudaMalloc((void**)&d_output_b, input_jpeg.width * input_jpeg.height * sizeof(unsigned char));

    // Copy input data from host to device
    cudaMemcpy(d_input_r, input_jpeg.r_values, input_jpeg.width * input_jpeg.height * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_input_g, input_jpeg.g_values, input_jpeg.width * input_jpeg.height * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_input_b, input_jpeg.b_values, input_jpeg.width * input_jpeg.height * sizeof(unsigned char), cudaMemcpyHostToDevice);

    // Set up grid and block dimensions
    dim3 block_dim(16, 16);
    dim3 grid_dim((input_jpeg.width + block_dim.x - 1) / block_dim.x, (input_jpeg.height + block_dim.y - 1) / block_dim.y);

    // Define bilateral filter parameters
    float sigma_spatial = 2.0f;
    float sigma_range = 30.0f;

    // Launch the kernel
    bilateral_filter<<<grid_dim, block_dim>>>(d_input_r, d_input_g, d_input_b, d_output_r, d_output_g, d_output_b,
                                              input_jpeg.width, input_jpeg.height, sigma_spatial, sigma_range);

    // Copy output data from device to host
    unsigned char* output_r = new unsigned char[input_jpeg.width * input_jpeg.height];
    unsigned char* output_g = new unsigned char[input_jpeg.width * input_jpeg.height];
    unsigned char* output_b = new unsigned char[input_jpeg.width * input_jpeg.height];
    cudaMemcpy(output_r, d_output_r, input_jpeg.width * input_jpeg.height * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    cudaMemcpy(output_g, d_output_g, input_jpeg.width * input_jpeg.height * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    cudaMemcpy(output_b, d_output_b, input_jpeg.width * input_jpeg.height * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    // Save output JPEG image
    const char* output_filename = argv[2];
    std::cout << "Output file to: " << output_filename << "\n";
    write_jpeg_soa(output_filename, input_jpeg.width, input_jpeg.height, output_r, output_g, output_b);

    // Clean up
    cudaFree(d_input_r);
    cudaFree(d_input_g);
    cudaFree(d_input_b);
    cudaFree(d_output_r);
    cudaFree(d_output_g);
    cudaFree(d_output_b);
    delete[] output_r;
    delete[] output_g;
    delete[] output_b;

    return 0;
}
