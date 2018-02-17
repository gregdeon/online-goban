# board_kernels.py
import numpy as np

# 0 = right
# 1 = up
# 2 = left
# 3 = down
# Assumes kernel width/height are same and even
def addDirection(kernel, direction):
    output = np.copy(kernel)
    size = len(kernel)
    mid = size // 2
    
    if direction == 0:
        output[mid-2:mid+2, mid-2:] = 0
        output[mid-1:mid+1, mid-1:] = 1
    elif direction == 1:
        output[:mid+2, mid-2:mid+2] = 0
        output[:mid+1, mid-1:mid+1] = 1
    elif direction == 2:
        output[mid-2:mid+2, :mid+2] = 0
        output[mid-1:mid+1, :mid+1] = 1
    elif direction == 3:
        output[mid-2:, mid-2:mid+2] = 0
        output[mid-1:, mid-1:mid+1] = 1
    
    return output

def getKernels(roi_size):
    base_kernel = np.full((roi_size, roi_size), -1)
    directions = [
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 0],
        [0, 1, 2],
        [1, 2, 3],
        [2, 3, 0],
        [3, 0, 1],
        [0, 1, 2, 3]
    ]
    
    output_list = []
    
    for dir_list in directions:
        kernel = np.copy(base_kernel)
        for d in dir_list:
            kernel = np.maximum(kernel, addDirection(kernel, d))
        output_list.append(kernel)
        
    return output_list
    
#print(getKernels(20))