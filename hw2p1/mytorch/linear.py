import numpy as np

class Linear:
    
    def __init__(self, in_features, out_features, debug = False):
    
        self.W    = np.zeros((out_features, in_features), dtype="f")
        self.b    = np.zeros((out_features, 1), dtype="f")
        self.dLdW = np.zeros((out_features, in_features), dtype="f")
        self.dLdb = np.zeros((out_features, 1), dtype="f")
        
        self.debug = debug

    def forward(self, A):
    
        self.A    = A
        self.N    = A.shape[0]
        self.Ones = np.ones((self.N,1), dtype="f")
        Z         = A.dot(self.W.transpose()) + self.Ones.dot(self.b.transpose())
        
        return Z
        
    def backward(self, dLdZ):
    
        dZdA      = self.W.transpose()
        dZdW      = self.A
        dZdi      = None
        dZdb      = self.Ones
        dLdA      = dLdZ.dot(dZdA.transpose())
        dLdW      = dLdZ.transpose().dot(dZdW)
        dLdi      = None
        dLdb      = dLdZ.transpose().dot(dZdb)
        self.dLdW = dLdW / self.N
        self.dLdb = dLdb / self.N

        if self.debug:
            
            self.dZdA = dZdA
            self.dZdW = dZdW
            self.dZdi = dZdi
            self.dZdb = dZdb
            self.dLdA = dLdA
            self.dLdi = dLdi
        
        return dLdA