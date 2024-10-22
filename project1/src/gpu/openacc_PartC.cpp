//
// Created by Zhong Yebin on 2023/9/16.
// Email: yebinzhong@link.cuhk.edu.cn
//
// OpenACC implementation of image filtering on JPEG
//

#include <memory.h>
#include <cstring>
#include <chrono>
#include <cmath>
#include <iostream>
#include <openacc.h>

#include "../utils.hpp"

#pragma acc routine seq
ColorValue acc_clamp_pixel_value(float value)
{
    return value > 255 ? 255
           : value < 0 ? 0
                       : static_cast<unsigned char>(value);
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
    const char* input_filename = argv[1];
    std::cout << "Input file from: " << input_filename << "\n";
    JpegSOA input_jpeg = read_jpeg_soa(input_filename);
    if (input_jpeg.r_values == nullptr)
    {
        std::cerr << "Failed to read input JPEG image\n";
        return -1;
    }


    int width = input_jpeg.width;
    int height = input_jpeg.height;
    int num_channels = input_jpeg.num_channels;
    
    JpegSOA output_jpeg(width, height, num_channels);
    
    #pragma acc parallel loop collapse(3) present(input_jpeg, output_jpeg)
    for (int c = 0; c < num_channels; c++)
    {
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                float sum = 0.0f;
                float weight_sum = 0.0f;
                float center_value = input_jpeg.get_pixel(x, y, c);
                
                for (int j = -1; j <= 1; j++)
                {
                    for (int i = -1; i <= 1; i++)
                    {
                        int neighbor_x = x + i;
                        int neighbor_y = y + j;
                        
                        if (neighbor_x >= 0 && neighbor_x < width && neighbor_y >= 0 && neighbor_y < height)
                        {
                            float neighbor_value = input_jpeg.get_pixel(neighbor_x, neighbor_y, c);
                            float spatial_distance = std::sqrt(i * i + j * j);
                            float intensity_distance = std::abs(center_value - neighbor_value);
                            float weight = std::exp(-spatial_distance / 2.0f) * std::exp(-intensity_distance / 2.0f);
                            
                            sum += neighbor_value * weight;
                            weight_sum += weight;
                        }
                    }
                }
                
                float filtered_value = sum / weight_sum;
                output_jpeg.set_pixel(x, y, c, filtered_value);
            }
        }
    }
    
    // Write JPEG File
    const char* output_filename = argv[2];
    std::cout << "Output file to: " << output_filename << "\n";
    write_jpeg_soa(output_filename, output_jpeg);
    
    return 0;
}
