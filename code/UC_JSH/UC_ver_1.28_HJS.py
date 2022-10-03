__author__ = "Yun-Su Kim, Jun-Hyeok Kim, Jinsol Hwang, Author2"
__copyright__ = "Copyright 2022, Gwangju Institute of Science and Technology"
__credits__ = ["Yun-Su Kim", "Jun-Hyeok Kim", "Jinsol Hwang", "Author2"]
__version__ = "1.1"
__maintainer__ = "Yun-Su Kim"
__email__ = "yunsukim@gist.ac.kr"
__status__ = "Test"

import numpy as np
import pandas as pd
from docplex.mp.model import Model
import UC_fig_v2
import UC_model
        
class MILP:
    def __init__(self, NAME, genList, nuclearList, pumpList, essList, load, UNIT_TIME, N_PIECE, bus, branch, flow_limit, nuclear_flag):

        self.m = Model(name=NAME)
        self.genList = genList
        self.nuclearList = nuclearList
        self.pumpList = pumpList
        self.essList = essList
        self.load = load
        self.UNIT_TIME = UNIT_TIME
        self.nGen = len(genList)
        self.nNuclear = len(nuclearList)
        self.nPump = len(pumpList)
        self.nEss = len(essList)
        self.nTimeslot = len(load)
        self.N_PIECE = N_PIECE
        self.bus = bus
        self.branch = branch
        self.nBus = np.shape(bus)[0]
        self.nBranch = np.shape(branch)[0]
        self.FLOW_LIMIT = flow_limit * np.ones(self.nBranch)
        self.B, _ = self.runDcPowerFlow(bus, branch)

        self.nuclear_flag = nuclear_flag# True - Nuclear Variable   False - Nuclear Sum

        ## var_Gen
        self.P_gen = self.m.continuous_var_cube(self.nGen, self.N_PIECE, self.nTimeslot, lb=0,
                                                ub=[self.genList[i].maxPowerPerPiece for i in range(self.nGen)
                                                    for _ in range(self.N_PIECE) for _ in range(self.nTimeslot)])
        # 1차 예비력
        self.P_genGF = self.m.continuous_var_matrix(self.nGen, self.nTimeslot, lb=0,
                                                    ub=[self.genList[i].GFRQ for i in range(self.nGen)
                                                        for _ in range(self.nTimeslot)])
        # 주파수제어예비력
        self.P_genAGC_FC = self.m.continuous_var_matrix(self.nGen, self.nTimeslot, lb=0,
                                                        ub=[self.genList[i].rampUpLimit*5 for i in range(self.nGen)
                                                            for _ in range(self.nTimeslot)])
        # 2차 예비력
        self.P_genAGC_sec = self.m.continuous_var_matrix(self.nGen, self.nTimeslot, lb=0,
                                                         ub=[self.genList[i].rampUpLimit*10 for i in range(self.nGen)
                                                             for _ in range(self.nTimeslot)])
        # 3차 예비력
        self.P_genSpin = self.m.continuous_var_matrix(self.nGen, self.nTimeslot, lb=0,
                                                      ub=[self.genList[i].rampUpLimit*30 for i in range(self.nGen)
                                                          for _ in range(self.nTimeslot)])
        # 속응성자원
        self.P_genNspin = self.m.continuous_var_matrix(self.nGen, self.nTimeslot, lb=0,
                                                       ub=[self.genList[i].AVAC for i in range(self.nGen)
                                                           for _ in range(self.nTimeslot)])
        self.U_gen = self.m.binary_var_matrix(self.nGen, self.nTimeslot)
        self.SU_gen = self.m.binary_var_matrix(self.nGen, self.nTimeslot)
        self.SD_gen = self.m.binary_var_matrix(self.nGen, self.nTimeslot)

        ## var_Nuclear
        self.P_nuclear = self.m.continuous_var_cube(self.nNuclear, self.N_PIECE, self.nTimeslot, lb=0,
                                                    ub=[self.nuclearList[i].maxPowerPerPiece for i in range(self.nNuclear)
                                                        for _ in range(self.N_PIECE) for _ in range(self.nTimeslot)])
        self.U_nuclear = self.m.binary_var_matrix(self.nNuclear, self.nTimeslot)
        self.SU_nuclear = self.m.binary_var_matrix(self.nNuclear, self.nTimeslot)
        self.SD_nuclear = self.m.binary_var_matrix(self.nNuclear, self.nTimeslot)

        ## var_Pump
        self.P_pumpChg = self.m.continuous_var_matrix(self.nPump, self.nTimeslot, lb=0,
                                                      ub=[self.pumpList[i].maxPump for i in range(self.nPump)
                                                          for _ in range(self.nTimeslot)])
        self.P_pumpDis = self.m.continuous_var_matrix(self.nPump, self.nTimeslot, lb=0,
                                                      ub=[self.pumpList[i].maxPower for i in range(self.nPump)
                                                          for _ in range(self.nTimeslot)])
        # 1차예비력
        self.P_pumpGF = self.m.continuous_var_matrix(self.nPump, self.nTimeslot, lb=0,
                                                     ub=[self.pumpList[i].GFRQ for i in range(self.nPump)
                                                         for _ in range(self.nTimeslot)])
        # 주파수제어예비력
        self.P_pumpAGC_FC = self.m.continuous_var_matrix(self.nPump, self.nTimeslot, lb=0,
                                                         ub=[self.pumpList[i].rampUpLimit*5 for i in range(self.nPump)
                                                             for _ in range(self.nTimeslot)])
        # 2차예비력
        self.P_pumpAGC_sec = self.m.continuous_var_matrix(self.nPump, self.nTimeslot, lb=0,
                                                          ub=[self.pumpList[i].rampUpLimit*10 for i in range(self.nPump)
                                                              for _ in range(self.nTimeslot)])
        # 3차예비력
        self.P_pumpSpin = self.m.continuous_var_matrix(self.nPump, self.nTimeslot, lb=0,
                                                       ub=[self.pumpList[i].rampUpLimit*30 for i in range(self.nPump)
                                                           for _ in range(self.nTimeslot)])
        # 속응성자원
        self.P_pumpNspin = self.m.continuous_var_matrix(self.nPump, self.nTimeslot, lb=0,
                                                        ub=[self.pumpList[i].AVAC for i in range(self.nPump)
                                                            for _ in range(self.nTimeslot)])
        self.U_pumpChg = self.m.binary_var_matrix(self.nPump, self.nTimeslot)
        self.U_pumpDis = self.m.binary_var_matrix(self.nPump, self.nTimeslot)

        ## var_ESS
        self.P_essChg = self.m.continuous_var_matrix(self.nEss, self.nTimeslot, lb=0,
                                                     ub=[self.essList[i].maxPower for i in range(self.nEss)
                                                         for _ in range(self.nTimeslot)])
        self.P_essDis = self.m.continuous_var_matrix(self.nEss, self.nTimeslot, lb=0,
                                                     ub=[self.essList[i].maxPower for i in range(self.nEss)
                                                         for _ in range(self.nTimeslot)])
        self.U_essChg = self.m.binary_var_matrix(self.nEss, self.nTimeslot)
        self.U_essDis = self.m.binary_var_matrix(self.nEss, self.nTimeslot)

    def constraint_minMax_balance(self):
        for j in range(self.nTimeslot):
            # balance
            self.m.add_constraint(sum(self.P_gen[i, k, j] for i in range(self.nGen) for k in range(self.N_PIECE))
                                  + sum(self.P_nuclear[i, k, j] for i in range(self.nNuclear) for k in range(self.N_PIECE))
                                  + sum(self.P_pumpDis[i, j] for i in range(self.nPump))
                                  - sum(self.P_pumpChg[i, j] for i in range(self.nPump))
                                  + sum(self.P_essDis[i, j] for i in range(self.nEss))
                                  - sum(self.P_essChg[i, j] for i in range(self.nEss)) == self.load[j])
            # Gen minmax
            for i in range(self.nGen):
                self.m.add_constraint(sum(self.P_gen[i, k, j] for k in range(self.N_PIECE))
                                      >= self.U_gen[i, j] * self.genList[i].minPower)
                self.m.add_constraint(sum(self.P_gen[i, k, j] for k in range(self.N_PIECE))
                                      <= self.U_gen[i, j] * self.genList[i].maxPower)
                self.m.add_constraint(self.SU_gen[i, j] + self.SD_gen[i, j] <= 1)

            # Nuclear minmax
            for i in range(self.nNuclear):
                if self.nuclear_flag:
                    self.m.add_constraint(sum(self.P_nuclear[i, k, j] for k in range(self.N_PIECE))
                                          >= self.U_nuclear[i, j] * self.nuclearList[i].minPower)
                    self.m.add_constraint(sum(self.P_nuclear[i, k, j] for k in range(self.N_PIECE))
                                          <= self.U_nuclear[i, j] * self.nuclearList[i].maxPower)
                    self.m.add_constraint(self.SU_nuclear[i, j] + self.SD_nuclear[i, j] <= 1)
                else:
                    self.m.add_constraint(self.U_nuclear[i, j] == 1)
                    self.m.add_constraints(self.P_nuclear[i, k, j] == self.nuclearList[i].maxPowerPerPiece
                                           for k in range(self.N_PIECE))
            # Pump minmax
            for i in range(self.nPump):
                self.m.add_constraint(self.P_pumpDis[i, j] <= self.U_pumpDis[i, j] * self.pumpList[i].maxPower)
                self.m.add_constraint(self.P_pumpDis[i, j] >= self.U_pumpDis[i, j] * self.pumpList[i].minPower)
                self.m.add_constraint(self.P_pumpChg[i, j] <= self.U_pumpChg[i, j] *
                                      (self.pumpList[i].maxPump - 0.9999 * self.pumpList[i].isFixedSpeed * (
                                              self.pumpList[i].maxPump - self.pumpList[i].fixedPumpPower)))
                self.m.add_constraint(self.P_pumpChg[i, j] >= self.U_pumpChg[i, j] *
                                      (self.pumpList[i].minPump + 0.9999 * self.pumpList[i].isFixedSpeed * (
                                              self.pumpList[i].fixedPumpPower - self.pumpList[i].minPump)))
                self.m.add_constraint(self.U_pumpDis[i, j] + self.U_pumpChg[i, j] <= 1)
            # Ess minmax
            for i in range(self.nEss):
                self.m.add_constraint(self.P_essDis[i, j] <= self.U_essDis[i, j] * self.essList[i].maxPower)
                self.m.add_constraint(self.P_essChg[i, j] <= self.U_essChg[i, j] * self.essList[i].maxPower)
                self.m.add_constraint(self.U_essDis[i, j] + self.U_essChg[i, j] <= 1)

        # 발전기 ONOFF 제약
        self.m.add_constraints(
            self.U_gen[i, j + 1] - self.U_gen[i, j] == self.SU_gen[i, j + 1] - self.SD_gen[i, j + 1]
            for i in range(self.nGen) for j in range(self.nTimeslot - 1))

        # 원자력 ONOFF 제약
        self.m.add_constraints(
            self.U_nuclear[i, j + 1] - self.U_nuclear[i, j] == self.SU_nuclear[i, j + 1] - self.SD_nuclear[i, j + 1]
            for i in range(self.nNuclear) for j in range(self.nTimeslot - 1))

    def constraint_SOC(self):
        for j in range(self.nTimeslot):
            # Pump SoC min max constraint
            for i in range(self.nPump):
                self.m.add_constraint(self.pumpList[i].initSOC
                                      - sum(self.P_pumpDis[i, k] * self.UNIT_TIME / self.pumpList[i].maxCapacity  # 양수는 발전시 효율 곱하지 않음
                                            for k in range(j + 1))
                                      + sum(self.P_pumpChg[i, k] * self.UNIT_TIME * self.pumpList[i].efficiency / self.pumpList[i].maxCapacity
                                            for k in range(j + 1)) <= self.pumpList[i].maxSOC)
                self.m.add_constraint(self.pumpList[i].initSOC
                                      - sum(self.P_pumpDis[i, k] * self.UNIT_TIME / self.pumpList[i].maxCapacity
                                            for k in range(j + 1))
                                      + sum(self.P_pumpChg[i, k] * self.UNIT_TIME * self.pumpList[i].efficiency / self.pumpList[i].maxCapacity
                                            for k in range(j + 1)) >= self.pumpList[i].minSOC)
            # Ess SoC min max constraint
            for i in range(self.nEss):
                self.m.add_constraint(self.essList[i].initSOC
                                      - sum(self.P_essDis[i, k] * self.UNIT_TIME / self.essList[i].efficiency / self.essList[i].maxCapacity
                                            for k in range(j + 1))
                                      + sum(self.P_essChg[i, k] * self.UNIT_TIME * self.essList[i].efficiency / self.essList[i].maxCapacity
                                            for k in range(j + 1)) <= self.essList[i].maxSOC)
                self.m.add_constraint(self.essList[i].initSOC
                                      - sum(self.P_essDis[i, k] * self.UNIT_TIME / self.essList[i].efficiency / self.essList[i].maxCapacity
                                            for k in range(j + 1))
                                      + sum(self.P_essChg[i, k] * self.UNIT_TIME * self.essList[i].efficiency / self.essList[i].maxCapacity
                                            for k in range(j + 1)) >= self.essList[i].minSOC)
        # Pump SoC terminal constraint
        for i in range(self.nPump):
            self.m.add_constraint(self.pumpList[i].initSOC
                                  - sum(self.P_pumpDis[i, k] * self.UNIT_TIME / self.pumpList[i].maxCapacity
                                        for k in range(self.nTimeslot))
                                  + sum(self.P_pumpChg[i, k] * self.UNIT_TIME * self.pumpList[i].efficiency / self.pumpList[i].maxCapacity
                                        for k in range(self.nTimeslot)) >= self.pumpList[i].termSOC)
        # Ess SoC terminal constraint
        for i in range(self.nEss):
            self.m.add_constraint(self.essList[i].initSOC
                                  - sum(self.P_essDis[i, k] * self.UNIT_TIME / self.essList[i].efficiency / self.essList[i].maxCapacity
                                        for k in range(self.nTimeslot))
                                  + sum(self.P_essChg[i, k] * self.UNIT_TIME * self.essList[i].efficiency / self.essList[i].maxCapacity
                                        for k in range(self.nTimeslot)) == self.essList[i].termSOC)

    def constraint_rampUpDown(self):
        for j in range(self.nTimeslot - 1):
            # 발전기 ramp up down 제약(reserve included)
            for i in range(self.nGen):

                self.m.add_constraint(sum(self.P_gen[i, k, j + 1] for k in range(self.N_PIECE))
                                      - sum(self.P_gen[i, k, j] for k in range(self.N_PIECE))
                                      + self.P_genGF[i, j] + self.P_genAGC_FC[i, j] + self.P_genAGC_sec[i, j] + self.P_genSpin[i, j]
                                      <= self.genList[i].rampUpLimit * 60 * self.UNIT_TIME)
                self.m.add_constraint(sum(self.P_gen[i, k, j + 1] for k in range(self.N_PIECE))
                                      - sum(self.P_gen[i, k, j] for k in range(self.N_PIECE))
                                      - self.P_genGF[i, j] - self.P_genAGC_FC[i, j] - self.P_genAGC_sec[i, j]
                                      >= -self.genList[i].rampDownLimit * 60 * self.UNIT_TIME)

            # 원자력 ramp up down 제약
            for i in range(self.nNuclear):
                self.m.add_constraint(sum(self.P_nuclear[i, k, j + 1] for k in range(self.N_PIECE))
                                      - sum(self.P_nuclear[i, k, j] for k in range(self.N_PIECE))
                                      <= self.nuclearList[i].rampUpLimit * 60 * self.UNIT_TIME)
                self.m.add_constraint(sum(self.P_nuclear[i, k, j + 1] for k in range(self.N_PIECE))
                                      - sum(self.P_nuclear[i, k, j] for k in range(self.N_PIECE))
                                      >= -self.nuclearList[i].rampDownLimit * 60 * self.UNIT_TIME)

            # 양수 ramp up down 제약(reserve included)
            for i in range(self.nPump):
                self.m.add_constraint(self.P_pumpDis[i, j + 1] - self.P_pumpDis[i, j]
                                      + self.P_pumpGF[i, j] + self.P_pumpAGC_FC[i, j] + self.P_pumpAGC_sec[i, j] + self.P_pumpSpin[i, j]
                                      <= self.pumpList[i].rampUpLimit * 60 * self.UNIT_TIME)
                self.m.add_constraint(self.P_pumpDis[i, j + 1] - self.P_pumpDis[i, j]
                                      - self.P_pumpGF[i, j] - self.P_pumpAGC_FC[i, j] - self.P_pumpAGC_sec[i, j]
                                      >= -self.pumpList[i].rampDownLimit * 60 * self.UNIT_TIME)
                # self.m.add_constraint( self.P_pumpChg[i, j + 1] - self.P_pumpChg[i, j] <= self.pumpList[i].rampUpLimit*60*self.UNIT_TIME )
                # self.m.add_constraint( self.P_pumpChg[i, j + 1] - self.P_pumpChg[i, j] >= -self.pumpList[i]rampDownLimit*60*self.UNIT_TIME )

    def constraint_gen_initState(self):
        for i in range(self.nGen):
            self.m.add_constraint(self.U_gen[i, 0] == self.SU_gen[i, 0])  # 시작할때부터 발전기 켜져있으면 startup 비용 추가

        for i in range(self.nNuclear):
            self.m.add_constraint(self.U_nuclear[i, 0] == self.SU_nuclear[i, 0])  # 시작할때부터 원자력 켜져있으면 startup 비용 추가

    def constraint_reserve(self, RESX=0.5):
        ## Gen reserve
        for i in range(self.nGen):
            for j in range(self.nTimeslot):
                ## GF constraints
                self.m.add_constraint(sum(self.P_gen[i, k, j] for k in range(self.N_PIECE)) + self.P_genGF[i, j]
                                      <= self.genList[i].AVAC * self.U_gen[i, j])
                self.m.add_constraint(sum(self.P_gen[i, k, j] for k in range(self.N_PIECE)) - self.P_genGF[i, j]
                                      >= self.genList[i].minPower * self.U_gen[i, j])
                # self.m.add_constraint(self.P_genGF[i, j]
                #                       == self.m.min(self.genList[i].maxGF - sum(self.P_gen[i, k, j] for k in range(self.N_PIECE)),
                #                                     self.genList[i].GFRQ,
                #                                     sum(self.P_gen[i, k, j] for k in range(self.N_PIECE)) - self.genList[i].minGF))

                ## AGC_FC constraints
                # upper bound
                self.m.add_constraint(sum(self.P_gen[i, k, j] for k in range(self.N_PIECE))
                                      + self.P_genGF[i, j] + self.P_genAGC_FC[i, j]
                                      <= self.genList[i].AVAC * self.U_gen[i, j])
                self.m.add_constraint(sum(self.P_gen[i, k, j] for k in range(self.N_PIECE)) + self.P_genAGC_FC[i, j]
                                      <= self.genList[i].maxLFC * self.U_gen[i, j])
                self.m.add_constraint(self.P_genAGC_FC[i, j] + self.P_genGF[i, j]
                                      <= self.genList[i].maxAGC * self.U_gen[i, j])
                # lower bound
                self.m.add_constraint(sum(self.P_gen[i, k, j] for k in range(self.N_PIECE))
                                      - self.P_genGF[i, j] - self.P_genAGC_FC[i, j]
                                      >= self.genList[i].minPower * self.U_gen[i, j])
                self.m.add_constraint(sum(self.P_gen[i, k, j] for k in range(self.N_PIECE)) - self.P_genAGC_FC[i, j]
                                      >= self.genList[i].minLFC * self.U_gen[i, j])
                # get AGC_FC
                # genAGC_FC_up = self.m.min(self.genList[i].maxLFC - sum(self.P_gen[i, k, j] for k in range(self.N_PIECE)),
                #                           5 * self.genList[i].rampUpLimit)
                # genAGC_FC_down = self.m.min(sum(self.P_gen[i, k, j] for k in range(self.N_PIECE)) - self.genList[i].minLFC,
                #                             5 * self.genList[i].rampDownLimit)
                # self.m.add_constraint(self.P_genAGC_FC[i, j] == self.m.min(genAGC_FC_up, genAGC_FC_down))

                ## AGC_sec constraints
                # upper bound
                self.m.add_constraint(sum(self.P_gen[i, k, j] for k in range(self.N_PIECE))
                                      + self.P_genGF[i, j] + self.P_genAGC_FC[i, j] + self.P_genAGC_sec[i, j]
                                      <= self.genList[i].AVAC * self.U_gen[i, j])
                self.m.add_constraint(sum(self.P_gen[i, k, j] for k in range(self.N_PIECE))
                                      + self.P_genAGC_FC[i, j] + self.P_genAGC_sec[i, j]
                                      <= self.genList[i].maxLFC * self.U_gen[i, j])
                self.m.add_constraint(self.P_genGF[i, j] + self.P_genAGC_FC[i, j] + self.P_genAGC_sec[i, j]
                                      <= self.genList[i].maxAGC * self.U_gen[i, j])
                self.m.add_constraint(self.P_genAGC_FC[i, j] + self.P_genAGC_sec[i, j]
                                      <= self.genList[i].maxAGC * self.U_gen[i, j])
                # lower bound
                self.m.add_constraint(sum(self.P_gen[i, k, j] for k in range(self.N_PIECE))
                                      - self.P_genGF[i, j] - self.P_genAGC_FC[i, j] - self.P_genAGC_sec[i, j]
                                      >= self.genList[i].minPower * self.U_gen[i, j])
                self.m.add_constraint(sum(self.P_gen[i, k, j] for k in range(self.N_PIECE))
                                      - self.P_genAGC_FC[i, j] - self.P_genAGC_sec[i, j]
                                      >= self.genList[i].minLFC * self.U_gen[i, j])
                # get AGC_sec
                # genAGC_sec_up = self.m.min(self.genList[i].maxLFC - sum(self.P_gen[i, k, j] for k in range(self.N_PIECE)),
                #                            10 * self.genList[i].rampUpLimit)
                # genAGC_sec_down = self.m.min(sum(self.P_gen[i, k, j] for k in range(self.N_PIECE)) - self.genList[i].minLFC,
                #                              10 * self.genList[i].rampDownLimit)
                # self.m.add_constraint(self.P_genAGC_sec[i, j] == self.m.min(genAGC_sec_up, genAGC_sec_down))

                ## Spinning constraints
                # 발전 중인 경우만 고려
                self.m.add_constraint(sum(self.P_gen[i, k, j] for k in range(self.N_PIECE)) + self.P_genGF[i, j] + self.P_genAGC_FC[i, j] + self.P_genAGC_sec[i, j] + self.P_genSpin[i, j]
                                      <= self.genList[i].AVAC * self.U_gen[i, j])
                # self.m.add_constraint(self.P_genSpin[i, j]
                #                       == self.m.min(self.genList[i].AVAC - sum(self.P_gen[i, k, j] for k in range(self.N_PIECE)),
                #                                     30 * self.genList[i].rampUpLimit))

                ## Nonspinning constraints
                # 정지 상태인 경우만 고려
                self.m.add_constraint(self.P_genNspin[i, j] <= self.genList[i].AVAC * (1 - self.U_gen[i, j]))
    
        ## Pump reserve
        for j in range(self.nTimeslot):
            for i in range(self.nPump):
                ## GF constraints
                self.m.add_constraint(self.P_pumpDis[i, j] + self.P_pumpGF[i, j]
                                      <= self.pumpList[i].AVAC * self.U_pumpDis[i, j])
                self.m.add_constraint(self.P_pumpDis[i, j] - self.P_pumpGF[i, j]
                                      >= self.pumpList[i].minPower * self.U_pumpDis[i, j])
                # self.m.add_constraint(self.P_pumpGF[i, j]
                #                       == self.m.min(self.pumpList[i].maxGF - self.P_pumpDis[i, j],
                #                                     self.pumpList[i].GFRQ,
                #                                     self.P_pumpDis[i, j] - self.pumpList[i].minGF))

                ## AGC_FC constraints
                # upper bound
                self.m.add_constraint(self.P_pumpDis[i, j] + self.P_pumpGF[i, j] + self.P_pumpAGC_FC[i, j]
                                      <= self.pumpList[i].AVAC * self.U_pumpDis[i, j])
                self.m.add_constraint(self.P_pumpDis[i, j] + self.P_pumpAGC_FC[i, j]
                                      <= self.pumpList[i].maxLFC * self.U_pumpDis[i, j])
                self.m.add_constraint(self.P_pumpAGC_FC[i, j] + self.P_pumpGF[i, j]
                                      <= self.pumpList[i].maxAGC * self.U_pumpDis[i, j])
                # lower bound
                self.m.add_constraint(self.P_pumpDis[i, j] - self.P_pumpGF[i, j] - self.P_pumpAGC_FC[i, j]
                                      >= self.pumpList[i].minPower * self.U_pumpDis[i, j])
                self.m.add_constraint(self.P_pumpDis[i, j] - self.P_pumpGF[i, j]
                                      >= self.pumpList[i].minLFC * self.U_pumpDis[i, j])
                # get AGC_FC
                # pumpAGC_FC_up = self.m.min(self.pumpList[i].maxLFC - self.P_pumpDis[i, j],
                #                            5 * self.pumpList[i].rampUpLimit)
                # pumpAGC_FC_down = self.m.min(self.P_pumpDis[i, j] - self.pumpList[i].minLFC,
                #                              5 * self.pumpList[i].rampDownLimit)
                # self.m.add_constraint(self.P_pumpAGC_FC[i, j] == self.m.min(pumpAGC_FC_up, pumpAGC_FC_down))

                ## AGC_sec constraints
                # upper bound
                self.m.add_constraint(self.P_pumpDis[i, j] + self.P_pumpGF[i, j]
                                      + self.P_pumpAGC_FC[i, j] + self.P_pumpAGC_sec[i, j]
                                      <= self.pumpList[i].AVAC * self.U_pumpDis[i, j])
                self.m.add_constraint(self.P_pumpDis[i, j] + self.P_pumpAGC_FC[i, j] + self.P_pumpAGC_sec[i, j]
                                      <= self.pumpList[i].maxLFC * self.U_pumpDis[i, j])
                self.m.add_constraint(self.P_pumpGF[i, j] + self.P_pumpAGC_FC[i, j] + self.P_pumpAGC_sec[i, j]
                                      <= self.pumpList[i].maxAGC * self.U_pumpDis[i, j])
                self.m.add_constraint(self.P_pumpAGC_FC[i, j] + self.P_pumpAGC_sec[i, j]
                                      <= self.pumpList[i].maxAGC * self.U_pumpDis[i, j])
                # lower bound
                self.m.add_constraint(self.P_pumpDis[i, j] - self.P_pumpGF[i, j]
                                      - self.P_pumpAGC_FC[i, j] - self.P_pumpAGC_sec[i, j]
                                      >= self.pumpList[i].minPower * self.U_pumpDis[i, j])
                self.m.add_constraint(self.P_pumpDis[i, j] - self.P_pumpAGC_FC[i, j] - self.P_pumpAGC_sec[i, j]
                                      >= self.pumpList[i].minLFC * self.U_pumpDis[i, j])
                # get AGC_sec
                # pumpAGC_sec_up = self.m.min(self.pumpList[i].maxLFC - self.P_pumpDis[i, j],
                #                             10 * self.pumpList[i].rampUpLimit)
                # pumpAGC_sec_down = self.m.min(self.P_pumpDis[i, j] - self.pumpList[i].minLFC,
                #                               10 * self.genList[i].rampDownLimit)
                # self.m.add_constraint(self.P_pumpAGC_sec[i, j] == self.m.min(pumpAGC_sec_up, pumpAGC_sec_down))

                ## Spinning constraints
                # 발전 중인 경우만 고려
                self.m.add_constraint(self.P_pumpDis[i, j] + self.P_pumpGF[i, j]
                                      + self.P_pumpAGC_FC[i, j] + self.P_pumpAGC_sec[i, j]
                                      + self.P_pumpSpin[i, j]
                                      <= self.pumpList[i].AVAC * self.U_pumpDis[i, j]
                                      + self.pumpList[i].maxPump * self.U_pumpChg[i, j])
                self.m.add_constraint(self.P_pumpSpin[i, j]
                                      <= self.pumpList[i].AVAC * self.U_pumpDis[i, j]
                                      + self.pumpList[i].maxPump * self.U_pumpChg[i, j])
                # self.m.add_constraint(self.P_pumpSpin[i, j]
                #                       == self.m.min(self.pumpList[i].AVAC - self.P_pumpDis[i, j],
                #                                     30 * self.pumpList[i].rampUpLimit))

                ## Nonspinning constraints
                self.m.add_constraint(self.P_pumpNspin[i, j] <= self.pumpList[i].AVAC * (1 - self.U_pumpDis[i, j]))
                self.m.add_constraint(self.P_pumpNspin[i, j] <= self.pumpList[i].AVAC * (1 - self.U_pumpChg[i, j]))

            ## Non-spinning reserve
            # self.m.add_constraint(sum(self.P_pumpNspin[i, j] for i in range(self.nPump))
            #                      <= RESX * sum(self.pumpList[i].maxCapacity for i in range(self.nPump)))

    def constraint_reserve_req(self, SCALE):
        [req_FC, req_GF, req_2, req_3, req_Nspin] = np.array([700, 1000, 1400, 1400, 2000]) / SCALE
        for j in range(self.nTimeslot):
            P_sum_GF = sum(self.P_genGF[i, j] for i in range(self.nGen)) \
                       + sum(self.P_pumpGF[i, j] for i in range(self.nPump))
            P_sum_AGC_FC = sum(self.P_genAGC_FC[i, j] for i in range(self.nGen)) \
                           + sum(self.P_pumpAGC_FC[i, j] for i in range(self.nPump))
            P_sum_AGC_sec = sum(self.P_genAGC_sec[i, j] for i in range(self.nGen)) \
                            + sum(self.P_pumpAGC_sec[i, j] for i in range(self.nPump))
            P_sum_Spin = sum(self.P_genSpin[i, j] for i in range(self.nGen)) \
                         + sum(self.P_pumpSpin[i, j] for i in range(self.nPump))
            P_sum_Nspin = sum(self.P_pumpNspin[i, j] for i in range(self.nPump))

            self.m.add_constraint(P_sum_GF >= req_GF)
            self.m.add_constraint(P_sum_AGC_FC >= req_FC)
            self.m.add_constraint(P_sum_AGC_FC + P_sum_AGC_sec >= req_FC + req_2)
            self.m.add_constraint(P_sum_AGC_FC + P_sum_AGC_sec + P_sum_Spin >= req_FC + req_2 + req_3)
            self.m.add_constraint(P_sum_GF + P_sum_AGC_FC + P_sum_AGC_sec + P_sum_Spin >= req_GF + req_FC + req_2 + req_3)
            self.m.add_constraint(P_sum_Nspin >= req_Nspin)

    def runDcPowerFlow(self, bus, branch):
        B = np.zeros((self.nBus, self.nBus))
        for i in range(self.nBranch):
            x = branch[i, 0, None]  # Get 'from bus'
            y = branch[i, 1, None]  # Get 'to bus'
            B[int(x - 1), int(y - 1)] = branch[i, 2, None]
            B[int(y - 1), int(x - 1)] = branch[i, 2, None]
            B[int(x - 1), int(x - 1)] = (B[int(x - 1), int(x - 1)] - branch[i, 2, None])
            B[int(y - 1), int(y - 1)] = (B[int(y - 1), int(y - 1)] - branch[i, 2, None])
        B = np.delete(B, 0, 0)  # Delete swing bus data
        B = np.delete(B, 0, 1)  # Delete swing bus data
        B = np.asmatrix(B)  # Convert to matrix
        B = np.linalg.inv(B)  # Invert Y-bus matrix

        busNetPower = bus[1:, 1, None] - bus[1:, 2, None]  # Solve for net power at each bus except swing bus
        busNetPower = np.asmatrix(busNetPower)  # Convert to matrix

        theta = B * busNetPower  # swing bus 제외한 theta 계산
        theta = np.vstack((0, theta))  # swing bus theta=0 row1에 추가

        output = branch.astype(float)  # Copy branch array over to an output array

        for i in range(self.nBranch):
            x = branch[i, 0, None]  # Get 'from bus'
            y = branch[i, 1, None]  # Get 'to bus'
            output[i, 2, None] = -branch[i, 2, None] * (theta[int(x - 1)] - theta[int(y - 1)])

        return B, output

    def cal_violation(self, violationList):
        nViolatedTimeslot = len(violationList)
        busNetPower_var = []
        theta_var = []
        for i in range(nViolatedTimeslot):
            t = int(violationList[i][0])  # 위배가 발생한 시간 index
            busNetPower_var.append([0] * (self.nBus - 1))
            theta_var.append([0] * (self.nBus - 1))

            # 전원들의 발전량을 위치한 모선에 할당
            # busNetPower_var = [0]*(nBus - 1)
            for k in range(self.nBus - 1):
                for l in range(self.nGen):
                    if self.genList[l].busNumber == k + 2:  # slack 모선을 제외했으므로 index 2가 0번째 row가 됨
                        busNetPower_var[i][k] = busNetPower_var[i][k] + sum(
                            self.P_gen[l, m, t] for m in range(self.N_PIECE))
                for l in range(self.nPump):
                    if self.pumpList[l].busNumber == k + 2:
                        busNetPower_var[i][k] = busNetPower_var[i][k] + self.P_pumpDis[l, t] - self.P_pumpChg[l, t]
                for l in range(self.nEss):
                    if self.essList[l].busNumber == k + 2:
                        busNetPower_var[i][k] = busNetPower_var[i][k] + self.P_essDis[l, t] - self.P_essChg[l, t]

            # 총부하량을 모선별 분배 할당 *** 향후 개선 필요
            for n in range(self.nBus - 1):
                busNetPower_var[i][n] = busNetPower_var[i][n] - self.load[int(t)] / (self.nBus - 1)
            busNetPower_var[i] = np.resize(busNetPower_var[i], (self.nBus - 1, 1))

            theta_var[i] = np.concatenate((np.zeros([1, 1]), self.B * busNetPower_var[i]), axis=0)

            nViolatedBranch = len(violationList[i]) - 1  # list에서 time index는 제외하므로 -1
            for j in range(nViolatedBranch):
                k = int(violationList[i][j + 1])  # 위배한 branch index
                self.m.add_constraint(-self.branch[k, 2, None][0] * (
                        theta_var[i][int(self.branch[k, 0, None] - 1), 0] - theta_var[i][
                    int(self.branch[k, 1, None] - 1), 0]) <= self.FLOW_LIMIT[k])
                self.m.add_constraint(-self.branch[k, 2, None][0] * (
                        theta_var[i][int(self.branch[k, 0, None] - 1), 0] - theta_var[i][
                    int(self.branch[k, 1, None] - 1), 0]) >= -self.FLOW_LIMIT[k])

    def check_flow_limit(self, violationList, sol):
        [P_genSol, P_pumpDisSol, P_pumpChgSol, P_essDisSol, P_essChgSol] = sol
        branch = np.loadtxt('branch.txt', delimiter=',', skiprows=1, dtype=float)
        cnt_violation = 0
        for i in range(self.nTimeslot):
            bus = np.loadtxt('bus.txt', delimiter=',', skiprows=1, dtype=float)
            for k in range(self.nBus - 1):
                for l in range(self.nGen):
                    if self.genList[l].busNumber == k + 2:  # slack 모선을 제외했으므로 index 2가 0번째 row가 됨
                        bus[k + 1, 1, None] = bus[k + 1, 1, None] + P_genSol[l, i]
                for l in range(self.nPump):
                    if self.pumpList[l].busNumber == k + 2:
                        bus[k + 1, 1, None] = bus[k + 1, 1, None] + P_pumpDisSol[l, i] - P_pumpChgSol[l, i]
                for l in range(self.nEss):
                    if self.essList[l].busNumber == k + 2:
                        bus[k + 1, 1, None] = bus[k + 1, 1, None] + P_essDisSol[l, i] - P_essChgSol[l, i]

            # 총부하량을 모선별 분배 할당 *** 향후 개선 필요
            for n in range(self.nBus - 1):
                bus[n + 1, 2, None] = bus[n + 1, 2, None] + self.load[i] / (self.nBus - 1)

            _, powerFlowResult = self.runDcPowerFlow(bus, branch)
            ## 제약위배가 발생하는 시간대, 발생한 branch만 표시
            if len(powerFlowResult[abs(powerFlowResult[:, 2]) > self.FLOW_LIMIT]) != 0:
                print('\nLine flow violation in time ' + str(i) + ':')
                print(powerFlowResult[abs(powerFlowResult[:, 2]) > self.FLOW_LIMIT])
                nViolatedBranch = np.shape(powerFlowResult[abs(powerFlowResult[:, 2]) > self.FLOW_LIMIT])[0]
                violatedBranch = np.zeros(nViolatedBranch)
                for j in range(nViolatedBranch):
                    violatedBranch[j] = np.where(abs(powerFlowResult[:, 2]) > self.FLOW_LIMIT)[0][j]
                violationList.append(np.concatenate((i * np.ones(1), violatedBranch)))
                cnt_violation += 1

        return violationList, cnt_violation

    def set_objective(self):
        self.m.set_objective("min", sum(self.genList[i].slopes[k] * self.P_gen[i, k, j] * self.genList[i].fuelCost
                                        for i in range(self.nGen) for k in range(self.N_PIECE) for j in range(self.nTimeslot)) * self.UNIT_TIME
                             + sum(self.U_gen[i, j] * self.genList[i].a * self.UNIT_TIME * self.genList[i].fuelCost
                                   + self.SU_gen[i, j] * self.genList[i].startupCost
                                   + self.SD_gen[i, j] * self.genList[i].shutdownCost for i in range(self.nGen) for j in range(self.nTimeslot))
                             + sum(self.nuclearList[i].slopes[k] * self.P_nuclear[i, k, j] * self.nuclearList[i].fuelCost for i in range(self.nNuclear)
                                   for k in range(self.N_PIECE) for j in range(self.nTimeslot)) * self.UNIT_TIME
                             + sum(self.U_nuclear[i, j] * self.nuclearList[i].a * self.UNIT_TIME * self.nuclearList[i].fuelCost
                                   + self.SU_nuclear[i, j] * self.nuclearList[i].startupCost
                                   + self.SD_nuclear[i, j] * self.nuclearList[i].shutdownCost for i in range(self.nNuclear) for j in range(self.nTimeslot)))

    def solve(self):
        self.m.print_information()
        new_sol = self.m.new_solution()
        #print(new_sol.find_unsatisfied_constraints(self.m))
        sol = self.m.solve()
        print(self.m.objective_value)

        return sol

    def get_sol(self, sol):
        ### Solution 저장
        P_genSol = np.zeros([self.nGen, self.nTimeslot])
        U_genSol = np.zeros([self.nGen, self.nTimeslot])
        SU_genSol = np.zeros([self.nGen, self.nTimeslot])
        SD_genSol = np.zeros([self.nGen, self.nTimeslot])

        P_nuclearSol = np.zeros([self.nNuclear, self.nTimeslot])
        U_nuclearSol = np.zeros([self.nNuclear, self.nTimeslot])
        SU_nuclearSol = np.zeros([self.nNuclear, self.nTimeslot])
        SD_nuclearSol = np.zeros([self.nNuclear, self.nTimeslot])

        P_pumpDisSol = np.zeros([self.nPump, self.nTimeslot])
        P_pumpChgSol = np.zeros([self.nPump, self.nTimeslot])
        U_pumpDisSol = np.zeros([self.nPump, self.nTimeslot])
        U_pumpChgSol = np.zeros([self.nPump, self.nTimeslot])
        P_essDisSol = np.zeros([self.nEss, self.nTimeslot])
        P_essChgSol = np.zeros([self.nEss, self.nTimeslot])
        U_essDisSol = np.zeros([self.nEss, self.nTimeslot])
        U_essChgSol = np.zeros([self.nEss, self.nTimeslot])
        socPump = np.zeros([self.nPump, self.nTimeslot])
        socEss = np.zeros([self.nEss, self.nTimeslot])

        for i in range(self.nTimeslot):
            for j in range(self.nGen):
                P_genSol[j, i] = sum(sol.get_value(self.P_gen[j, k, i]) for k in range(self.N_PIECE))
                U_genSol[j, i] = sol.get_value(self.U_gen[j, i])
                SU_genSol[j, i] = sol.get_value(self.SU_gen[j, i])
                SD_genSol[j, i] = sol.get_value(self.SD_gen[j, i])

            for j in range(self.nNuclear):
                P_nuclearSol[j, i] = sum(sol.get_value(self.P_nuclear[j, k, i]) for k in range(self.N_PIECE))
                U_nuclearSol[j, i] = sol.get_value(self.U_nuclear[j, i])
                SU_nuclearSol[j, i] = sol.get_value(self.SU_nuclear[j, i])
                SD_nuclearSol[j, i] = sol.get_value(self.SD_nuclear[j, i])

            for j in range(self.nPump):
                P_pumpDisSol[j, i] = sol.get_value(self.P_pumpDis[j, i])
                P_pumpChgSol[j, i] = sol.get_value(self.P_pumpChg[j, i])
                U_pumpDisSol[j, i] = sol.get_value(self.U_pumpDis[j, i])
                U_pumpChgSol[j, i] = sol.get_value(self.U_pumpChg[j, i])
                socPump[j, i] = (self.pumpList[j].initSOC - sum(
                    P_pumpDisSol[j, k] * self.UNIT_TIME / self.pumpList[j].efficiency / self.pumpList[j].maxCapacity for
                    k in range(i + 1))
                                 + sum(
                            P_pumpChgSol[j, k] * self.UNIT_TIME * self.pumpList[j].efficiency / self.pumpList[
                                j].maxCapacity for k in range(i + 1)))
            for j in range(self.nEss):
                P_essDisSol[j, i] = sol.get_value(self.P_essDis[j, i])
                P_essChgSol[j, i] = sol.get_value(self.P_essChg[j, i])
                U_essDisSol[j, i] = sol.get_value(self.U_essDis[j, i])
                U_essChgSol[j, i] = sol.get_value(self.U_essChg[j, i])
                socEss[j, i] = (self.essList[j].initSOC - sum(
                    P_essDisSol[j, k] * self.UNIT_TIME / self.essList[j].efficiency / self.essList[j].maxCapacity for k
                    in range(i + 1)) + sum(
                    P_essChgSol[j, k] * self.UNIT_TIME * self.essList[j].efficiency / self.essList[j].maxCapacity for k
                    in range(i + 1)))

        return [P_genSol, P_nuclearSol, P_pumpDisSol, P_pumpChgSol, P_essDisSol, P_essChgSol], \
               [U_genSol, SU_genSol, SD_genSol, U_nuclearSol, SU_nuclearSol, SD_nuclearSol,
                U_pumpDisSol, U_pumpChgSol, U_essDisSol, U_essChgSol], \
               [socPump, socEss]

    def get_sol_reserve(self, sol):
        ## reserve solution
        P_genGF = np.zeros((self.nGen, self.nTimeslot))
        P_genAGC_FC = np.zeros((self.nGen, self.nTimeslot))
        P_genAGC_sec = np.zeros((self.nGen, self.nTimeslot))
        P_genSpin = np.zeros((self.nGen, self.nTimeslot))
        P_genNspin = np.zeros((self.nGen, self.nTimeslot))

        P_pumpGF = np.zeros((self.nPump, self.nTimeslot))
        P_pumpAGC_FC = np.zeros((self.nPump, self.nTimeslot))
        P_pumpAGC_sec = np.zeros((self.nPump, self.nTimeslot))
        P_pumpSpin = np.zeros((self.nPump, self.nTimeslot))
        P_pumpNspin = np.zeros((self.nPump, self.nTimeslot))

        for j in range(self.nTimeslot):
            for i in range(self.nGen):
                P_genGF[i, j] = sol.get_value(self.P_genGF[i, j])
                P_genAGC_FC[i, j] = sol.get_value(self.P_genAGC_FC[i, j])
                P_genAGC_sec[i, j] = sol.get_value(self.P_genAGC_sec[i, j])
                P_genSpin[i, j] = sol.get_value(self.P_genSpin[i, j])
                P_genNspin[i, j] = sol.get_value(self.P_genNspin[i, j])
            for i in range(self.nPump):
                P_pumpGF[i, j] = sol.get_value(self.P_pumpGF[i, j])
                P_pumpAGC_FC[i, j] = sol.get_value(self.P_pumpAGC_FC[i, j])
                P_pumpAGC_sec[i, j] = sol.get_value(self.P_pumpAGC_sec[i, j])
                P_pumpSpin[i, j] = sol.get_value(self.P_pumpSpin[i, j])
                P_pumpNspin[i, j] = sol.get_value(self.P_pumpNspin[i, j])

        return [P_genGF, P_pumpGF], [P_genAGC_FC, P_pumpAGC_FC], [P_genAGC_sec, P_pumpAGC_sec], \
               [P_genSpin, P_pumpSpin], [P_genNspin, P_pumpNspin]


if __name__ == "__main__":

    genCodeList = [4401, 2900, 2637, 7032, 7012]
    #genCodeList = 'All'
    pumpCodeList = [1610, 1630]#, 1620, 1650, 1640]  # , '화천수력', '춘천수력']
    #pumpCodeList = 'All'

    loadPattern = pd.read_excel("./data/수요예측내역(22_6).xlsx", 0, header=3)
    dload = (loadPattern['최대'] - loadPattern['최소'])[:-2]
    repLoad = loadPattern.loc[dload.idxmax()]
    load = repLoad[1:25]  # 1시 - 24시
    if genCodeList == 'All':
        SCALE = 1
    else:
        SCALE = 30

    load = load / SCALE
    UNIT_TIME = 1  # Timeslot마다 에너지MWh를 구하기 위해 출력MW에 곱해져야하는 시간 단위, 1/4 = 15 min
    pumpSOC = {'SOC_PUMP_MIN': 0.5, 'SOC_PUMP_MAX': 1.0, 'SOC_PUMP_INIT': 0.9, 'SOC_PUMP_TERM': 0.9}
    essSOC = {'SOC_ESS_MIN': 0.3, 'SOC_ESS_MAX': 0.8, 'SOC_ESS_INIT': 0.5, 'SOC_ESS_TERM': 0.5}

    N_PIECE = 3  # cost function 선형화 구간 수
    bus = np.loadtxt('bus.txt', delimiter=',', skiprows=1, dtype=float)
    branch = np.loadtxt('branch.txt', delimiter=',', skiprows=1, dtype=float)

    ess1 = {'name': 'ESS1', 'busNumber': 3, 'minPower': 0, 'maxPower': 100, 'maxCapacity': 500,
            'efficiency': 0.8, 'minSOC': essSOC['SOC_ESS_MIN'], 'maxSOC': essSOC['SOC_ESS_MAX'],
            'initSOC': essSOC['SOC_ESS_INIT'], 'termSOC': essSOC['SOC_ESS_TERM']}
    
    ess2 = {'name': 'ESS2', 'busNumber': 3, 'minPower': 0, 'maxPower': 200, 'maxCapacity': 800,
            'efficiency': 0.8, 'minSOC': essSOC['SOC_ESS_MIN'], 'maxSOC': essSOC['SOC_ESS_MAX'],
            'initSOC': essSOC['SOC_ESS_INIT'], 'termSOC': essSOC['SOC_ESS_TERM']}
    
    essGroup = [ess1, ess2]

    modelList = UC_model.GetModel(N_PIECE, genCodeList, pumpCodeList, pumpSOC, essGroup)
    genList, nuclearList, pumpList, essList = modelList.ReadModel()

    # for i in range(len(pumpList)):
    #     pumpList[i].minPower = 0
    pumpList[0].fixedPumpPower = 60
    pumpList[1].fixedPumpPower = 40

    modelList.printModelparameters(genList, nuclearList, pumpList, essList)

    violationList = []
    stop_flag = False
    # while not stop_flag:
    #     UC = MILP('Unit Commitment', genList, nuclearList, pumpList, essList, load, UNIT_TIME, N_PIECE, bus, branch,
    #               flow_limit=300, nuclear_flag=True)
    #     UC.constraint_minMax_balance()
    #     UC.constraint_SOC()
    #     UC.constraint_rampUpDown()
    #     UC.constraint_reserve()
    #     UC.constraint_reserve_req(SCALE)
    #     UC.constraint_gen_initState()
    #     # UC.cal_violation(violationList)
    #     UC.set_objective()
    #     sol = UC.solve()
    #     P_sol, U_sol, SoC_sol = UC.get_sol(sol)
    #     GF_sol, AGC_FC_sol, AGC_sec_sol, Spin_sol, Nspin_sol = UC.get_sol_reserve(sol)
    #     fig = UC_fig_v2.UC_plot(UNIT_TIME, genList, nuclearList, pumpList, essList, load)
    #     # fig.make_plot(P_sol, U_sol, SoC_sol)
    #     fig.make_plot_new(P_sol, SoC_sol)
    #     fig.make_res_plot(GF_sol, AGC_FC_sol, AGC_sec_sol, Spin_sol, Nspin_sol)
    #     fig.make_res_cascading(GF_sol, AGC_FC_sol, AGC_sec_sol, Spin_sol, Nspin_sol, SCALE, True)
    #     stop_flag = True

        # violationList, cnt_violation = UC.check_flow_limit(violationList, P_sol)
        # print(violationList)
        # if cnt_violation == 0:
        #     stop_flag = True
        #     fig.make_plot_new(P_sol, SoC_sol)
