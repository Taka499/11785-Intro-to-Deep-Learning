import numpy as np


class Identity:
    
    def forward(self, Z):
    
        self.A = Z
        
        return self.A
    
    def backward(self):
    
        dAdZ = np.ones(self.A.shape, dtype="f")
        
        return dAdZ


class Sigmoid:
    
    def forward(self, Z):
        ## sigmoid(z) = 1 / (1 + exp(-z))
        self.A = np.divide(1, 1 + np.exp(-1*Z))
        
        return self.A
    
    def backward(self):
        ## sigmoid backward = A - A**2
        dAdZ = self.A - self.A**2
        
        return dAdZ


class Tanh:
    
    def forward(self, Z):
        ## Tanh = tanh(Z) = sinh(Z) / cosh(Z)
        self.A = np.tanh(Z)
        
        return self.A
    
    def backward(self):
        ## Tanh backward = 1 - A**2
        dAdZ = 1 - self.A**2
        
        return dAdZ


class ReLU:
    
    def forward(self, Z):
        ## ReLU(Z) = max(Z, 0)
        self.A = np.maximum(Z, 0)
        
        return self.A
    
    def backward(self):
        ## ReLU backward = 1 if A>0 else 0
        dAdZ = np.where(self.A > 0, 1, 0)
        
        return dAdZ
        
        
