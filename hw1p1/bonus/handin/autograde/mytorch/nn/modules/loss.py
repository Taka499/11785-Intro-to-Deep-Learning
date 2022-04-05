import numpy as np

class MSELoss:
    
    def forward(self, A, Y):
    
        self.A = A
        self.Y = Y
        N      = A.shape[0]
        C      = A.shape[1]
        ## se = hadamard product(element wise matrix multiplication) of A-Y
        ## ---- that is: (A-Y)**2 element wise
        se     = np.multiply((A-Y), (A-Y))
        ## sse = sum of squared error
        ##     = ones_N.transpose (dot) se (dot) ones_C
        sse    = np.dot( np.dot(np.ones(N, dtype="f"), se), np.ones((C, 1), dtype="f") )
        mse    = sse/(N*C)
        
        return mse
    
    def backward(self):
    
        dLdA = self.A - self.Y
        
        return dLdA

class CrossEntropyLoss:
    
    def forward(self, A, Y):
    
        self.A   = A
        self.Y   = Y
        N        = A.shape[0]
        C        = A.shape[1]
        Ones_C   = np.ones((C, 1), dtype="f")
        Ones_N   = np.ones((N, 1), dtype="f")

        ## softmax = exp(x_i) / sum(exp(all x_i)) --> require element wise division on exp(A), divided by the sum of each row of exp(A)
        self.softmax     = np.divide(np.exp(A), np.exp(A).dot(Ones_C).dot(Ones_C.transpose()))
        ## cross_entropy = -Y element wise multiply log(softmax)
        crossentropy     = -1 * np.multiply(Y, np.log(self.softmax))
        sum_crossentropy = Ones_N.transpose().dot(crossentropy).dot(Ones_C)
        L = sum_crossentropy / N
        
        return L
    
    def backward(self):
    
        dLdA = self.softmax - self.Y
        
        return dLdA
