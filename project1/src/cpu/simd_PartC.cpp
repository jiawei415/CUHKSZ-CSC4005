//
// Created by Liu Yuxuan on 2024/9/10
// Modified on Yang Yufan's simd_PartB.cpp on 2023/9/16
// Email: yufanyang1@link.cuhk.edu.cn
// Email: yuxuanliu1@link.cuhk.edu.cn
//
// SIMD (AVX2) implementation of transferring a JPEG picture from RGB to gray
//

#include <memory.h>
#include <chrono>
#include <iostream>
#include <immintrin.h>

#include "../utils.hpp"

#include <immintrin.h>

// Function to perform Bilateral Filtering using SIMD (AVX2)
void bilateralFilterSIMD(__m256i* src, __m256i* dst, int width, int height, float sigma_spatial, float sigma_range)
{
    // Calculate constants for spatial and range Gaussian kernels
    float spatial_coeff = -0.5f / (sigma_spatial * sigma_spatial);
    float range_coeff = -0.5f / (sigma_range * sigma_range);
    
    // Loop over each pixel in the image
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x += 8)
        {
            // Load 8 pixels from source image
            __m256i src_pixels = _mm256_loadu_si256(src + y * width + x);
            
            // Initialize sum and weight accumulators
            __m256 sum = _mm256_setzero_ps();
            __m256 weight_sum = _mm256_setzero_ps();
            
            // Loop over the 8 pixels
            for (int i = 0; i < 8; i++)
            {
                // Extract the RGB values of the current pixel
                __m256i rgb = _mm256_and_si256(src_pixels, _mm256_set1_epi32(0xFF));
                
                // Calculate the spatial and range distances
                __m256i spatial_dist = _mm256_set1_epi32((x + i) * (x + i) + y * y);
                __m256i range_dist = _mm256_sub_epi32(rgb, _mm256_set1_epi32(128));
                range_dist = _mm256_mullo_epi32(range_dist, range_dist);
                
                // Calculate the spatial and range weights
                __m256 spatial_weight = _mm256_exp_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(spatial_dist), _mm256_set1_ps(spatial_coeff)));
                __m256 range_weight = _mm256_exp_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(range_dist), _mm256_set1_ps(range_coeff)));
                
                // Calculate the final weight
                __m256 weight = _mm256_mul_ps(spatial_weight, range_weight);
                
                // Accumulate the weighted sum and weight
                sum = _mm256_fmadd_ps(_mm256_cvtepi32_ps(rgb), weight, sum);
                weight_sum = _mm256_add_ps(weight, weight_sum);
                
                // Shift to the next pixel
                src_pixels = _mm256_srli_epi32(src_pixels, 8);
            }
            
            // Normalize the sum by the weight
            __m256 result = _mm256_div_ps(sum, weight_sum);
            
            // Store the result in the destination image
            _mm256_storeu_si256(dst + y * width + x, _mm256_cvtps_epi32(result));
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
    // Read JPEG File
    const char* input_filepath = argv[1];
    std::cout << "Input file from: " << input_filepath << "\n";
    JpegSOA input_jpeg = read_jpeg_soa(input_filepath);
    if (input_jpeg.r_values == nullptr)
    {
        std::cerr << "Failed to read input JPEG image\n";
        return -1;
    }
    auto start_time = std::chrono::high_resolution_clock::now();

    int width = input_jpeg.width;
    int height = input_jpeg.height;
    float sigma_spatial = 1.0f;
    float sigma_range = 10.0f;
    
    // Allocate memory for the destination image
    JpegSOA output_jpeg;
    output_jpeg.width = width;
    output_jpeg.height = height;
    output_jpeg.r_values = new uint8_t[width * height];
    output_jpeg.g_values = new uint8_t[width * height];
    output_jpeg.b_values = new uint8_t[width * height];
    
    // Perform Bilateral Filtering using SIMD
    bilateralFilterSIMD((__m256i*)input_jpeg.r_values, (__m256i*)output_jpeg.r_values, width, height, sigma_spatial, sigma_range);
    bilateralFilterSIMD((__m256i*)input_jpeg.g_values, (__m256i*)output_jpeg.g_values, width, height, sigma_spatial, sigma_range);
    bilateralFilterSIMD((__m256i*)input_jpeg.b_values, (__m256i*)output_jpeg.b_values, width, height, sigma_spatial, sigma_range);
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time);
    std::cout << "Transformation Complete!" << std::endl;
    std::cout << "Execution Time: " << elapsed_time.count()
              << " milliseconds\n";
    return 0;
}
