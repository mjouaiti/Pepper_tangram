#coding=utf-8
#!/usr/bin/env python2

'''
    MIT License
    
    Copyright (c) 2019 Nicolas Descouens
    
    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:
    
    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.
    
    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
'''

from math import *
import numpy as np

class LowPassFilter:
    def __init__(self, K, f_c):
        self.K = K
        self.omega_c = (2 * np.pi * f_c)

    def step(self, x, y, deltaT):
        return float(self.K * x + y / (self.omega_c * deltaT)) / (1 + 1 / (self.omega_c * deltaT))

    # def step(self, x, y, deltaT):
    #     return float((self.K * self.omega_c * (x - y) + y) / (1 + self.omega_c * (1 + self.K) + 1))

class Integrator:
    def __init__(self, K, T):
        self.K = K
        self.T = T
        
    def step(self, x, y, deltaT):
        return float(self.K * deltaT * x / self.T + y)

tm = 1.

def getSpeed(currentTheta, commandTheta):
    return float(commandTheta - currentTheta) / tm


class HighPassFilter:
    def __init__(self, K, f_c):
        self.K = K
        self.omega_c = (2 * np.pi * f_c)
        
    def step(self, x, y, deltaT):
        return float(self.K * x + y / (self.omega_c * deltaT)) / (1 + 1 / (self.omega_c * deltaT))
    
