import numpy as np
from docplex.mp.model import Model

### 발전기 객체 모델 / 단위: Power in MW, Ramp in MW/min
class ThermalGenerator:
    def __init__(self, name, minMaxPower, costCoeff, numberOfPieces):
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