import torch
import torch.nn.functional as F
import triton
import triton.language as tl
import numpy as np
import cv2

@triton.jit
def bilateral_filter(input, output, width, height, ksize, sigma_space, sigma_density):
    # Define the kernel function for bilateral filtering
    @triton.jit
    def kernel(i, j):
        sum_w = 0.0
        sum_val = 0.0
        center_val = input[i, j]
        for x in range(-ksize // 2, ksize // 2 + 1):
            for y in range(-ksize // 2, ksize // 2 + 1):
                if i + x >= 0 and i + x < height and j + y >= 0 and j + y < width:
                    neighbor_val = input[i + x, j + y]
                    w = tl.exp(-(x * x + y * y) / (2 * sigma_space * sigma_space))
                    w *= tl.exp(-(center_val - neighbor_val) * (center_val - neighbor_val) / (2 * sigma_density * sigma_density))
                    sum_w += w
                    sum_val += w * neighbor_val
        output[i, j] = sum_val / sum_w

    # Launch the kernel function using Triton
    triton.kernel(kernel)(height, width)

def main(input_image_path, output_image_path):
    sigma_space = 1.7
    sigma_density = 50.0
    ksize = 7


    # Load the input image
    image = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)

    # Convert the image to a torch tensor
    input_tensor = torch.from_numpy(image.astype(np.float32))

    # Create an output tensor with the same shape as the input tensor
    output_tensor = torch.zeros_like(input_tensor)

    # Call the bilateral_filter function
    bilateral_filter(input_tensor, output_tensor, image.shape[1], image.shape[0],
                     ksize, sigma_space, sigma_density)

    # Convert the output tensor back to a numpy array
    output_image = output_tensor.numpy().astype(np.uint8)

    # Save the output image
    cv2.imwrite(output_image_path, output_image)

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Invalid argument, should be: python3 triton_PartC.py /path/to/input/jpeg /path/to/output/jpeg")
        sys.exit(-1)
    main(sys.argv[1], sys.argv[2])
