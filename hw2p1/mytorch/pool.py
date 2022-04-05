import numpy as np
from resampling import *

class MaxPool2d_stride1():

    def __init__(self, kernel):
        self.kernel = kernel

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_height, input_width)
        Return:
            Z (np.array): (batch_size, out_channels, output_height, output_width)
        """

        output_height = (A.shape[2] - self.kernel) + 1
        output_width = (A.shape[3] - self.kernel) + 1
        self.max_indices = np.zeros((A.shape[0], A.shape[1], output_height, output_width, 4), dtype=int)
        self.shape = A.shape

        Z = np.zeros((A.shape[0], A.shape[1], output_height, output_width))
        for x in range(output_height):
            for y in range(output_width):
                Z[:, :, x, y] = np.amax(A[:, :, x:x+self.kernel, y:y+self.kernel], axis=(2, 3))
                indices = np.argmax(A[:, :, x:x+self.kernel, y:y+self.kernel].reshape(A.shape[0], A.shape[1], -1), axis=2)
                self.max_indices[:, :, x, y, 0] = (x + indices//self.kernel)
                self.max_indices[:, :, x, y, 1] = (y + indices%self.kernel)

        return Z
    
    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_height, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_height, input_width)
        """

        dLdA = np.zeros(self.shape)
        # TODO how to optimize this?
        for b in range(self.shape[0]):
            for c in range(self.shape[1]):
                for x in range(self.max_indices.shape[2]):
                    for y in range(self.max_indices.shape[3]):
                        dLdA[b, c, self.max_indices[b, c, x, y, 0], self.max_indices[b, c, x, y, 1]] += dLdZ[b, c, x, y]

        return dLdA

class MeanPool2d_stride1():

    def __init__(self, kernel):
        self.kernel = kernel

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """

        self.shape = A.shape
        output_height = (self.shape[2] - self.kernel) + 1
        output_width = (self.shape[3] - self.kernel) + 1

        Z = np.zeros((self.shape[0], self.shape[1], output_height, output_width))
        for x in range(output_height):
            for y in range(output_width):
                Z[:, :, x, y] = np.mean(A[:, :, x:x+self.kernel, y:y+self.kernel], axis=(2, 3))

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """

        dLdZ_pad = np.pad(dLdZ, pad_width=((0,0), (0,0), (self.kernel-1, self.kernel-1), (self.kernel-1, self.kernel-1)), mode='constant', constant_values=0)
        dLdA = np.zeros(self.shape)
        for m in range(self.shape[2]):
            for n in range(self.shape[3]):
                dLdA[:, :, m, n] = np.mean(dLdZ_pad[:, :, m:m+self.kernel, n:n+self.kernel], axis=(2, 3))

        return dLdA

class MaxPool2d():

    def __init__(self, kernel, stride):
        self.kernel = kernel
        self.stride = stride
        
        #Create an instance of MaxPool2d_stride1
        self.maxpool2d_stride1 = MaxPool2d_stride1(kernel=kernel)
        self.downsample2d = Downsample2d(downsampling_factor=stride)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """

        Z_stride1 = self.maxpool2d_stride1.forward(A)
        Z = self.downsample2d.forward(Z_stride1)
        
        return Z
    
    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        
        dLdZ_stride1 = self.downsample2d.backward(dLdZ)
        dLdA = self.maxpool2d_stride1.backward(dLdZ_stride1)
        
        return dLdA

class MeanPool2d():

    def __init__(self, kernel, stride):
        self.kernel = kernel
        self.stride = stride

        #Create an instance of MaxPool2d_stride1
        self.meanpool2d_stride1 = MeanPool2d_stride1(kernel=kernel)
        self.downsample2d = Downsample2d(downsampling_factor=stride)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """

        Z_stride1 = self.meanpool2d_stride1.forward(A)
        Z = self.downsample2d.forward(Z_stride1)

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """

        dLdZ_stride1 = self.downsample2d.backward(dLdZ)
        dLdA = self.meanpool2d_stride1.backward(dLdZ_stride1)

        return dLdA
