import numpy as np

class Upsample1d():

    def __init__(self, upsampling_factor):
        self.upsampling_factor = upsampling_factor

    def forward(self, A):

        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width)
        Return:
            Z (np.array): (batch_size, in_channels, output_width)
        """

        N = A.shape[0]
        C = A.shape[1]
        W0 = A.shape[2]

        Z = np.zeros((N, C, (W0 - 1) * self.upsampling_factor + 1))
        for n in range(N):
            for c in range(C):
                for w in range(W0):
                    Z[n, c, w*self.upsampling_factor] = A[n, c, w]

        return Z

    def backward(self, dLdZ):

        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width)
        """
        
        N = dLdZ.shape[0]
        C = dLdZ.shape[1]
        W1 = dLdZ.shape[2]

        dLdA = np.zeros((N, C, (W1 - 1) // self.upsampling_factor + 1))
        for n in range(N):
            for c in range(C):
                for w in range(0, W1, self.upsampling_factor):
                    dLdA[n, c, w//self.upsampling_factor] = dLdZ[n, c, w]

        return dLdA

class Downsample1d():

    def __init__(self, downsampling_factor):
        self.downsampling_factor = downsampling_factor

    def forward(self, A):

        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width)
        Return:
            Z (np.array): (batch_size, in_channels, output_width)
        """

        N = A.shape[0]
        C = A.shape[1]
        self.W0 = A.shape[2]

        Z = np.zeros((N, C, (self.W0 - 1) // self.downsampling_factor + 1))
        for n in range(N):
            for c in range(C):
                for w in range(0, self.W0, self.downsampling_factor):
                    Z[n, c, w//self.downsampling_factor] = A[n, c, w]

        return Z

    def backward(self, dLdZ):

        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width)
        """
        
        N = dLdZ.shape[0]
        C = dLdZ.shape[1]
        W1 = dLdZ.shape[2]

        dLdA = np.zeros((N, C, self.W0))
        for n in range(N):
            for c in range(C):
                for w in range(W1):
                    dLdA[n, c, w*self.downsampling_factor] = dLdZ[n, c, w]

        return dLdA

class Upsample2d():

    def __init__(self, upsampling_factor):
        self.upsampling_factor = upsampling_factor

    def forward(self, A):

        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, in_channels, output_width, output_height)
        """

        N = A.shape[0]
        C = A.shape[1]
        H0 = A.shape[2]
        W0 = A.shape[3]
        H1 = (H0 - 1) * self.upsampling_factor + 1
        W1 = (W0 - 1) * self.upsampling_factor + 1

        Z = np.zeros((N, C, H1, W1))
        for n in range(N):
            for c in range(C):
                for h in range(H0):
                    for w in range(W0):
                        Z[n, c, h*self.upsampling_factor, w*self.upsampling_factor] = A[n, c, h, w]

        return Z

    def backward(self, dLdZ):

        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """

        N = dLdZ.shape[0]
        C = dLdZ.shape[1]
        H1 = dLdZ.shape[2]
        W1 = dLdZ.shape[3]
        H0 = (H1 - 1) // self.upsampling_factor + 1
        W0 = (W1 - 1) // self.upsampling_factor + 1

        dLdA = np.zeros((N, C, H0, W0))
        for n in range(N):
            for c in range(C):
                for h in range(0, H1, self.upsampling_factor):
                    for w in range(0, W1, self.upsampling_factor):
                        dLdA[n, c, h//self.upsampling_factor, w//self.upsampling_factor] = dLdZ[n, c, h, w]

        return dLdA

class Downsample2d():

    def __init__(self, downsampling_factor):
        self.downsampling_factor = downsampling_factor

    def forward(self, A):

        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, in_channels, output_width, output_height)
        """

        N = A.shape[0]
        C = A.shape[1]
        self.H0 = A.shape[2]
        self.W0 = A.shape[3]
        H1 = (self.H0 - 1) // self.downsampling_factor + 1
        W1 = (self.W0 - 1) // self.downsampling_factor + 1

        Z = np.zeros((N, C, H1, W1))
        for n in range(N):
            for c in range(C):
                for h in range(0, self.H0, self.downsampling_factor):
                    for w in range(0, self.W0, self.downsampling_factor):
                        Z[n, c, h//self.downsampling_factor, w//self.downsampling_factor] = A[n, c, h, w]

        return Z

    def backward(self, dLdZ):

        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        
        N = dLdZ.shape[0]
        C = dLdZ.shape[1]
        H1 = dLdZ.shape[2]
        W1 = dLdZ.shape[3]

        dLdA = np.zeros((N, C, self.H0, self.W0))
        for n in range(N):
            for c in range(C):
                for h in range(H1):
                    for w in range(W1):
                        dLdA[n, c, h*self.downsampling_factor, w*self.downsampling_factor] = dLdZ[n, c, h, w]

        return dLdA