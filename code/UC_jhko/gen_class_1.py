import numpy as np
import matplotlib.pyplot as plt
from docplex.mp.model import Model

### 발전기 객체 모델 / 단위: Power in MW, Ramp in MW/min
class ThermalGenerator:
    def __init__(self, name, minMaxPower,  costCoeff, numberOfPieces):
        self.name = name
        self.minPower = minMaxPower[0]
        self.maxPower = minMaxPower[1]
        self.a = costCoeff[0]
        self.startupCost = costCoeff[3]
        self.shutdownCost = costCoeff[4]
        
        slopesOfCostFunction = np.zeros(numberOfPieces)
        maxPowerPerPiece = self.maxPower / numberOfPieces
        self.maxPowerPerPiece = maxPowerPerPiece
        for i in range(numberOfPieces):
            slopesOfCostFunction[i] = ( costCoeff[0] + costCoeff[1]*(maxPowerPerPiece*(i + 1)) 
                                       + costCoeff[2]*(maxPowerPerPiece*(i + 1))**2
                                       - costCoeff[0] - costCoeff[1]*maxPowerPerPiece*i
                                       - costCoeff[2]*(maxPowerPerPiece*i)**2 ) / maxPowerPerPiece
        self.slopes = slopesOfCostFunction

        
class EnergyStorage:
    def __init__(self, name, minMaxPower, initTermSOC, maxCapacity, efficiency, minMaxSOC):
        self.name = name
        self.minPower = minMaxPower[0]
        self.maxPower = minMaxPower[1]
        self.initSOC = initTermSOC[0]
        self.termSOC = initTermSOC[1]
        self.maxCapacity = maxCapacity
        self.minSOC = maxCapacity*minMaxSOC[0]
        self.maxSOC = maxCapacity*minMaxSOC[1]
        self.chgEff = efficiency[0]
        self.disEff = efficiency[1]

### 함수 목록
def calculateGenCost(slopes, powers, isOn, a):
    nSlope = len(slopes)
    cost = 0
    for i in range(nSlope):
        cost = cost + slopes[i]*powers[i]
    cost = cost + isOn*a
    return cost