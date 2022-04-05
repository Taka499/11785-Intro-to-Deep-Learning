# Do not import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)

import numpy as np
from resampling import *

class Conv1d_stride1():
    def __init__(self, in_channels, out_channels, kernel_size,
                 weight_init_fn=None, bias_init_fn=None):
        # Do not modify this method
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        if weight_init_fn is None:
            self.W = np.random.normal(0, 1.0, (out_channels, in_channels, kernel_size))
        else:
            self.W = weight_init_fn(out_channels, in_channels, kernel_size)

        if bias_init_fn is None:
            self.b = np.zeros(out_channels)
        else:
            self.b = bias_init_fn(out_channels)

        self.dLdW = np.zeros(self.W.shape)
        self.dLdb = np.zeros(self.b.shape)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_size)
        Return:
            Z (np.array): (batch_size, out_channels, output_size)
        """
        self.A = A
        output_size = self.A.shape[2] - self.kernel_size + 1

        Z = np.zeros((self.A.shape[0], self.out_channels, output_size))
        for i in range(output_size):
            Z[:, :, i] = np.tensordot(self.A[:, :, i:i+self.kernel_size], self.W, axes=([1, 2], [1, 2])) + self.b
        
        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_size)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_size)
        """

        for k in range(self.kernel_size):
            self.dLdW[:, :, k] = np.tensordot(dLdZ, self.A[:, :, k:k+dLdZ.shape[2]], axes=((0,2), (0,2)))
        self.dLdb = np.sum(dLdZ, axis=(0,2))

        dLdZ_pad = np.pad(dLdZ, pad_width=((0, 0), (0, 0), (self.kernel_size-1, self.kernel_size-1)), constant_values=0, mode='constant')
        W_flipped = np.flip(self.W, axis=2)
        dLdA = np.zeros(self.A.shape)
        for i in range(self.A.shape[2]):
            dLdA[:, :, i] = np.tensordot(dLdZ_pad[:, :, i:i+self.kernel_size], W_flipped, axes=((1, 2), (0, 2)))

        return dLdA

class Conv1d():
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                weight_init_fn=None, bias_init_fn=None):
        # Do not modify the variable names
    
        self.stride = stride

        # Initialize Conv1d() and Downsample1d() isntance
        self.conv1d_stride1 = Conv1d_stride1(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, weight_init_fn=weight_init_fn, bias_init_fn=bias_init_fn)
        self.downsample1d = Downsample1d(self.stride)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_size)
        Return:
            Z (np.array): (batch_size, out_channels, output_size)
        """

        # Call Conv1d_stride1
        Z_stride1 = self.conv1d_stride1.forward(A)

        # downsample
        Z = self.downsample1d.forward(Z_stride1)

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_size)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_size)
        """
        # Call downsample1d backward
        dLdZ_stride1 = self.downsample1d.backward(dLdZ)

        # Call Conv1d_stride1 backward
        dLdA = self.conv1d_stride1.backward(dLdZ_stride1) 

        return dLdA


class Conv2d_stride1():
    def __init__(self, in_channels, out_channels,
                 kernel_size, weight_init_fn=None, bias_init_fn=None):

        # Do not modify this method

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        if weight_init_fn is None:
            self.W = np.random.normal(0, 1.0, (out_channels, in_channels, kernel_size, kernel_size))
        else:
            self.W = weight_init_fn(out_channels, in_channels, kernel_size, kernel_size)

        if bias_init_fn is None:
            self.b = np.zeros(out_channels)
        else:
            self.b = bias_init_fn(out_channels)

        self.dLdW = np.zeros(self.W.shape)
        self.dLdb = np.zeros(self.b.shape)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_height, input_width)
        Return:
            Z (np.array): (batch_size, out_channels, output_height, output_width)
        """
        self.A = A
        output_height = (self.A.shape[2] - self.kernel_size) + 1
        output_width = (self.A.shape[3] - self.kernel_size) + 1

        Z = np.zeros((self.A.shape[0], self.out_channels, output_height, output_width))
        for x in range(output_height):
            for y in range(output_width):
                Z[:, :, x, y] = np.tensordot(A[:, :, x:x+self.kernel_size, y:y+self.kernel_size], self.W, axes=((1, 2, 3), (1, 2, 3)))

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_height, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_height, input_width)
        """

        for i in range(self.kernel_size):
            for j in range(self.kernel_size):
                self.dLdW[:, :, i, j] = np.tensordot(dLdZ, self.A[:, :, i:i+dLdZ.shape[2], j:j+dLdZ.shape[3]], axes=((0,2,3), (0,2,3)))
        self.dLdb = np.sum(dLdZ, axis=(0, 2, 3))

        dLdZ_pad = np.pad(dLdZ, pad_width=((0, 0), (0, 0), (self.kernel_size-1, self.kernel_size-1), (self.kernel_size-1, self.kernel_size-1)), constant_values=0, mode='constant')
        W_flipped = np.flip(self.W, axis=(2, 3))
        dLdA = np.zeros((self.A.shape))
        for m in range(self.A.shape[2]):
            for n in range(self.A.shape[3]):
                dLdA[:, :, m, n] = np.tensordot(dLdZ_pad[:, :, m:m+self.kernel_size, n:n+self.kernel_size], W_flipped, axes=((1, 2, 3), (0, 2, 3)))

        return dLdA

class Conv2d():
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                weight_init_fn=None, bias_init_fn=None):
        # Do not modify the variable names
        self.stride = stride

        # Initialize Conv2d() and Downsample2d() isntance
        self.conv2d_stride1 = Conv2d_stride1(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, weight_init_fn=weight_init_fn, bias_init_fn=bias_init_fn)
        self.downsample2d = Downsample2d(self.stride)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        # Call Conv2d_stride1
        Z_stride1 = self.conv2d_stride1.forward(A)

        # downsample
        Z = self.downsample2d.forward(Z_stride1)

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """

        # Call downsample1d backward
        dLdZ_stride1 = self.downsample2d.backward(dLdZ)

        # Call Conv1d_stride1 backward
        dLdA = self.conv2d_stride1.backward(dLdZ_stride1)

        return dLdA

class ConvTranspose1d():
    def __init__(self, in_channels, out_channels, kernel_size, upsampling_factor,
                weight_init_fn=None, bias_init_fn=None):
        # Do not modify this method
        self.upsampling_factor = upsampling_factor

        # Initialize Conv1d stride 1 and upsample1d isntance
        self.upsample1d = Upsample1d(self.upsampling_factor)
        self.conv1d_stride1 = Conv1d_stride1(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, weight_init_fn=weight_init_fn, bias_init_fn=bias_init_fn)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_size)
        Return:
            Z (np.array): (batch_size, out_channels, output_size)
        """

        # upsample
        A_upsampled = self.upsample1d.forward(A)

        # Call Conv1d_stride1()
        Z = self.conv1d_stride1.forward(A_upsampled)

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_size)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_size)
        """

        #Call backward in the correct order
        delta_out = self.conv1d_stride1.backward(dLdZ)

        dLdA = self.upsample1d.backward(delta_out)

        return dLdA

class ConvTranspose2d():
    def __init__(self, in_channels, out_channels, kernel_size, upsampling_factor,
                weight_init_fn=None, bias_init_fn=None):
        # Do not modify this method
        self.upsampling_factor = upsampling_factor

        # Initialize Conv2d() isntance
        self.conv2d_stride1 = Conv2d_stride1(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, weight_init_fn=weight_init_fn, bias_init_fn=bias_init_fn)
        self.upsample2d = Upsample2d(self.upsampling_factor)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_size)
        Return:
            Z (np.array): (batch_size, out_channels, output_size)
        """
        # upsample
        A_upsampled = self.upsample2d.forward(A)

        # Call Conv2d_stride1()
        Z = self.conv2d_stride1.forward(A_upsampled)

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_size)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_size)
        """
        #Call backward in correct order
        delta_out = self.conv2d_stride1.backward(dLdZ)

        dLdA = self.upsample2d.backward(delta_out)

        return dLdA

class Flatten():

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, in_width)
        Return:
            Z (np.array): (batch_size, in_channels * in width)
        """

        self.shape = A.shape
        Z = A.reshape(A.shape[0], -1)

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch size, in channels * in width)
        Return:
            dLdA (np.array): (batch size, in channels, in width)
        """

        dLdA = dLdZ.reshape(self.shape)

        return dLdA

