# -*- coding: utf-8 -*-
"""
Created on Mon Sep  5 11:39:38 2022

@author: junhyeok
"""
import pandas as pd
from docplex.mp.model import Model

import numpy as np
import time
import math

# MILP Object
class MILP:
    def __init__(self, NAME, modelDict, load, UNIT_TIME, N_PIECE, bus, branch, flow_limit, nuclear_flag):

        self.m = Model(name=NAME)
        self.genList = modelDict['gen']
        self.nuclearList = modelDict['nuclear']
        self.pumpList = modelDict['pump']
        self.essList = modelDict['ess']
        self.load = load
        self.UNIT_TIME = UNIT_TIME

        self.nGen = len(self.genList)
        self.nNuclear = len(self.nuclearList)
        self.nPump = len(self.pumpList)
        self.nEss = len(self.essList)
        self.nTimeslot = len(load)

        self.N_PIECE = N_PIECE
        self.bus = bus
        self.branch = branch
        self.nBus = np.shape(bus)[0]
        self.nBranch = np.shape(branch)[0]
        self.FLOW_LIMIT = flow_limit * np.ones(self.nBranch)
        self.B, _ = self.runDcPowerFlow(bus, branch)

        self.nuclear_flag = nuclear_flag  # True - Nuclear Variable   False - Nuclear Sum

        # var_Gen
        self.P_gen = self.m.continuous_var_cube(self.nGen, self.N_PIECE, self.nTimeslot, lb=0,
                                                ub=[self.genList[i].maxPowerPerPiece for i in range(self.nGen)
                                                    for _ in range(self.N_PIECE) for _ in range(self.nTimeslot)])
        ## 1차 예비력
        self.P_gen_pri = self.m.continuous_var_matrix(self.nGen, self.nTimeslot, lb=0,
                                                      ub=[self.genList[i].GFRQ
                                                          if not math.isnan(self.genList[i].GFRQ)
                                                          else self.genList[i].maxGF
                                                          for i in range(self.nGen)
                                                          for _ in range(self.nTimeslot)])
        ## 주파수제어예비력
        self.P_gen_reg = self.m.continuous_var_matrix(self.nGen, self.nTimeslot, lb=0,
                                                      ub=[self.genList[i].rampUpLimit * 5 for i in range(self.nGen)
                                                          for _ in range(self.nTimeslot)])
        ## 2차 예비력
        self.P_gen_sec = self.m.continuous_var_matrix(self.nGen, self.nTimeslot, lb=0,
                                                      ub=[self.genList[i].rampUpLimit * 10 for i in range(self.nGen)
                                                          for _ in range(self.nTimeslot)])
        ## 3차 예비력(운전)
        self.P_gen_ter = self.m.continuous_var_matrix(self.nGen, self.nTimeslot, lb=0,
                                                      ub=[self.genList[i].rampUpLimit * 30 for i in range(self.nGen)
                                                          for _ in range(self.nTimeslot)])
        ## 속응성자원
        self.P_gen_rap = self.m.continuous_var_matrix(self.nGen, self.nTimeslot, lb=0,
                                                      ub=[self.genList[i].AVAC for i in range(self.nGen)
                                                          for _ in range(self.nTimeslot)])

        self.U_gen = self.m.binary_var_matrix(self.nGen, self.nTimeslot)
        self.SU_gen = self.m.binary_var_matrix(self.nGen, self.nTimeslot)
        self.SD_gen = self.m.binary_var_matrix(self.nGen, self.nTimeslot)

        # var_Nuclear
        self.P_nuclear = self.m.continuous_var_cube(self.nNuclear, self.N_PIECE, self.nTimeslot, lb=0,
                                                    ub=[self.nuclearList[i].maxPowerPerPiece for i in
                                                        range(self.nNuclear)
                                                        for _ in range(self.N_PIECE) for _ in range(self.nTimeslot)])
        self.U_nuclear = self.m.binary_var_matrix(self.nNuclear, self.nTimeslot)
        self.SU_nuclear = self.m.binary_var_matrix(self.nNuclear, self.nTimeslot)
        self.SD_nuclear = self.m.binary_var_matrix(self.nNuclear, self.nTimeslot)

        # var_Pump
        self.P_pumpChg = self.m.continuous_var_matrix(self.nPump, self.nTimeslot, lb=0,
                                                      ub=[self.pumpList[i].maxPump for i in range(self.nPump)
                                                          for _ in range(self.nTimeslot)])
        self.P_pumpDis = self.m.continuous_var_matrix(self.nPump, self.nTimeslot, lb=0,
                                                      ub=[self.pumpList[i].maxPower for i in range(self.nPump)
                                                          for _ in range(self.nTimeslot)])
        ## 1차예비력
        self.P_pump_pri = self.m.continuous_var_matrix(self.nPump, self.nTimeslot, lb=0,
                                                       ub=[self.pumpList[i].GFRQ for i in range(self.nPump)
                                                           for _ in range(self.nTimeslot)])
        ## 주파수제어예비력
        self.P_pump_reg = self.m.continuous_var_matrix(self.nPump, self.nTimeslot, lb=0,
                                                       ub=[self.pumpList[i].rampUpLimit * 5 for i in range(self.nPump)
                                                           for _ in range(self.nTimeslot)])
        ## 2차예비력
        self.P_pump_sec = self.m.continuous_var_matrix(self.nPump, self.nTimeslot, lb=0,
                                                       ub=[self.pumpList[i].rampUpLimit * 10 for i in range(self.nPump)
                                                           for _ in range(self.nTimeslot)])
        ## 3차예비력(운전)
        self.P_pump_terSpin = self.m.continuous_var_matrix(self.nPump, self.nTimeslot, lb=0,
                                                           ub=[self.pumpList[i].rampUpLimit * 30 for i in range(self.nPump)
                                                               for _ in range(self.nTimeslot)])
        ## 3차예비력(정지)
        self.P_pump_terNspin = self.m.continuous_var_matrix(self.nPump, self.nTimeslot, lb=0,
                                                            ub=[self.pumpList[i].AVAC for i in range(self.nPump)
                                                                for _ in range(self.nTimeslot)])
        ## 속응성자원
        self.P_pump_rap = self.m.continuous_var_matrix(self.nPump, self.nTimeslot, lb=0,
                                                       ub=[self.pumpList[i].AVAC for i in range(self.nPump)
                                                           for _ in range(self.nTimeslot)])

        self.U_pumpChg = self.m.binary_var_matrix(self.nPump, self.nTimeslot)
        self.U_pumpDis = self.m.binary_var_matrix(self.nPump, self.nTimeslot)

        # var_ESS
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
                                      + self.P_gen_pri[i, j] + self.P_gen_reg[i, j] + self.P_gen_sec[i, j] + self.P_gen_ter[i, j]
                                      <= self.genList[i].rampUpLimit * 60 * self.UNIT_TIME)
                self.m.add_constraint(sum(self.P_gen[i, k, j + 1] for k in range(self.N_PIECE))
                                      - sum(self.P_gen[i, k, j] for k in range(self.N_PIECE))
                                      - self.P_gen_pri[i, j] - self.P_gen_reg[i, j] - self.P_gen_sec[i, j]
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
                                      + self.P_pump_pri[i, j] + self.P_pump_reg[i, j] + self.P_pump_sec[i, j] + self.P_pump_terSpin[i, j]
                                      <= self.pumpList[i].rampUpLimit * 60 * self.UNIT_TIME)
                self.m.add_constraint(self.P_pumpDis[i, j + 1] - self.P_pumpDis[i, j]
                                      - self.P_pump_pri[i, j] - self.P_pump_reg[i, j] - self.P_pump_sec[i, j]
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
                ## Primary constraints
                self.m.add_constraint(sum(self.P_gen[i, k, j] for k in range(self.N_PIECE)) + self.P_gen_pri[i, j]
                                      <= self.genList[i].maxGF * self.U_gen[i, j])
                self.m.add_constraint(sum(self.P_gen[i, k, j] for k in range(self.N_PIECE)) - self.P_gen_pri[i, j]
                                      >= self.genList[i].minPower * self.U_gen[i, j])
                # self.m.add_constraint(self.P_genGF[i, j]
                #                       == self.m.min(self.genList[i].maxGF - sum(self.P_gen[i, k, j] for k in range(self.N_PIECE)),
                #                                     self.genList[i].GFRQ,
                #                                     sum(self.P_gen[i, k, j] for k in range(self.N_PIECE)) - self.genList[i].minGF))

                ## Frequency regulation constraints
                # upper bound
                self.m.add_constraint(sum(self.P_gen[i, k, j] for k in range(self.N_PIECE))
                                      + self.P_gen_pri[i, j] + self.P_gen_reg[i, j]
                                      <= self.genList[i].AVAC * self.U_gen[i, j])
                self.m.add_constraint(sum(self.P_gen[i, k, j] for k in range(self.N_PIECE)) + self.P_gen_reg[i, j]
                                      <= self.genList[i].maxLFC * self.U_gen[i, j])
                self.m.add_constraint(self.P_gen_reg[i, j] + self.P_gen_pri[i, j]
                                      <= self.genList[i].maxAGC * self.U_gen[i, j])
                # lower bound
                self.m.add_constraint(sum(self.P_gen[i, k, j] for k in range(self.N_PIECE))
                                      - self.P_gen_pri[i, j] - self.P_gen_reg[i, j]
                                      >= self.genList[i].minPower * self.U_gen[i, j])
                self.m.add_constraint(sum(self.P_gen[i, k, j] for k in range(self.N_PIECE)) - self.P_gen_reg[i, j]
                                      >= self.genList[i].minLFC * self.U_gen[i, j])
                # get AGC_FC
                # genAGC_FC_up = self.m.min(self.genList[i].maxLFC - sum(self.P_gen[i, k, j] for k in range(self.N_PIECE)),
                #                           5 * self.genList[i].rampUpLimit)
                # genAGC_FC_down = self.m.min(sum(self.P_gen[i, k, j] for k in range(self.N_PIECE)) - self.genList[i].minLFC,
                #                             5 * self.genList[i].rampDownLimit)
                # self.m.add_constraint(self.P_genAGC_FC[i, j] == self.m.min(genAGC_FC_up, genAGC_FC_down))

                ## Secondary constraints
                # upper bound
                self.m.add_constraint(sum(self.P_gen[i, k, j] for k in range(self.N_PIECE))
                                      + self.P_gen_pri[i, j] + self.P_gen_reg[i, j] + self.P_gen_sec[i, j]
                                      <= self.genList[i].AVAC * self.U_gen[i, j])
                self.m.add_constraint(sum(self.P_gen[i, k, j] for k in range(self.N_PIECE))
                                      + self.P_gen_reg[i, j] + self.P_gen_sec[i, j]
                                      <= self.genList[i].maxLFC * self.U_gen[i, j])
                self.m.add_constraint(self.P_gen_pri[i, j] + self.P_gen_reg[i, j] + self.P_gen_sec[i, j]
                                      <= self.genList[i].maxAGC * self.U_gen[i, j])
                # self.m.add_constraint(self.P_genAGC_FC[i, j] + self.P_genAGC_sec[i, j]
                #                       <= self.genList[i].maxAGC * self.U_gen[i, j])
                # lower bound
                self.m.add_constraint(sum(self.P_gen[i, k, j] for k in range(self.N_PIECE))
                                      - self.P_gen_pri[i, j] - self.P_gen_reg[i, j] - self.P_gen_sec[i, j]
                                      >= self.genList[i].minPower * self.U_gen[i, j])
                self.m.add_constraint(sum(self.P_gen[i, k, j] for k in range(self.N_PIECE))
                                      - self.P_gen_reg[i, j] - self.P_gen_sec[i, j]
                                      >= self.genList[i].minLFC * self.U_gen[i, j])
                # get AGC_sec
                # genAGC_sec_up = self.m.min(self.genList[i].maxLFC - sum(self.P_gen[i, k, j] for k in range(self.N_PIECE)),
                #                            10 * self.genList[i].rampUpLimit)
                # genAGC_sec_down = self.m.min(sum(self.P_gen[i, k, j] for k in range(self.N_PIECE)) - self.genList[i].minLFC,
                #                              10 * self.genList[i].rampDownLimit)
                # self.m.add_constraint(self.P_genAGC_sec[i, j] == self.m.min(genAGC_sec_up, genAGC_sec_down))

                ## Tertiary constraints
                # 발전 중인 경우만 고려
                self.m.add_constraint(sum(self.P_gen[i, k, j] for k in range(self.N_PIECE))
                                      + self.P_gen_pri[i, j] + self.P_gen_reg[i, j] + self.P_gen_sec[i, j] + self.P_gen_ter[i, j]
                                      <= self.genList[i].AVAC * self.U_gen[i, j])
                # self.m.add_constraint(self.P_genSpin[i, j]
                #                       == self.m.min(self.genList[i].AVAC - sum(self.P_gen[i, k, j] for k in range(self.N_PIECE)),
                #                                     30 * self.genList[i].rampUpLimit))

                ## Rapid(속응성) constraints
                # 정지 상태인 경우만 고려
                self.m.add_constraint(self.P_gen_rap[i, j] <= self.genList[i].AVAC * (1 - self.U_gen[i, j]))

        ## Pump reserve
        for j in range(self.nTimeslot):
            for i in range(self.nPump):
                ## Primary constraints
                self.m.add_constraint(self.P_pumpDis[i, j] + self.P_pump_pri[i, j]
                                      <= self.pumpList[i].maxGF * self.U_pumpDis[i, j])
                self.m.add_constraint(self.P_pumpDis[i, j] - self.P_pump_pri[i, j]
                                      >= self.pumpList[i].minPower * self.U_pumpDis[i, j])
                # self.m.add_constraint(self.P_pumpGF[i, j]
                #                       == self.m.min(self.pumpList[i].maxGF - self.P_pumpDis[i, j],
                #                                     self.pumpList[i].GFRQ,
                #                                     self.P_pumpDis[i, j] - self.pumpList[i].minGF))

                ## Frequency regulation constraints
                # upper bound
                self.m.add_constraint(self.P_pumpDis[i, j] + self.P_pump_pri[i, j] + self.P_pump_reg[i, j]
                                      <= self.pumpList[i].AVAC * self.U_pumpDis[i, j])
                self.m.add_constraint(self.P_pumpDis[i, j] + self.P_pump_reg[i, j]
                                      <= self.pumpList[i].maxLFC * self.U_pumpDis[i, j])
                self.m.add_constraint(self.P_pump_reg[i, j] + self.P_pump_pri[i, j]
                                      <= self.pumpList[i].maxAGC * self.U_pumpDis[i, j])
                # lower bound
                self.m.add_constraint(self.P_pumpDis[i, j] - self.P_pump_pri[i, j] - self.P_pump_reg[i, j]
                                      >= self.pumpList[i].minPower * self.U_pumpDis[i, j])
                self.m.add_constraint(self.P_pumpDis[i, j] - self.P_pump_pri[i, j]
                                      >= self.pumpList[i].minLFC * self.U_pumpDis[i, j])
                # get AGC_FC
                # pumpAGC_FC_up = self.m.min(self.pumpList[i].maxLFC - self.P_pumpDis[i, j],
                #                            5 * self.pumpList[i].rampUpLimit)
                # pumpAGC_FC_down = self.m.min(self.P_pumpDis[i, j] - self.pumpList[i].minLFC,
                #                              5 * self.pumpList[i].rampDownLimit)
                # self.m.add_constraint(self.P_pumpAGC_FC[i, j] == self.m.min(pumpAGC_FC_up, pumpAGC_FC_down))

                ## Secondary constraints
                # upper bound
                self.m.add_constraint(self.P_pumpDis[i, j] + self.P_pump_pri[i, j]
                                      + self.P_pump_reg[i, j] + self.P_pump_sec[i, j]
                                      <= self.pumpList[i].AVAC * self.U_pumpDis[i, j])
                self.m.add_constraint(self.P_pumpDis[i, j] + self.P_pump_reg[i, j] + self.P_pump_sec[i, j]
                                      <= self.pumpList[i].maxLFC * self.U_pumpDis[i, j])
                self.m.add_constraint(self.P_pump_pri[i, j] + self.P_pump_reg[i, j] + self.P_pump_sec[i, j]
                                      <= self.pumpList[i].maxAGC * self.U_pumpDis[i, j])
                # self.m.add_constraint(self.P_pumpAGC_FC[i, j] + self.P_pumpAGC_sec[i, j]
                #                       <= self.pumpList[i].maxAGC * self.U_pumpDis[i, j])
                # lower bound
                self.m.add_constraint(self.P_pumpDis[i, j] - self.P_pump_pri[i, j]
                                      - self.P_pump_reg[i, j] - self.P_pump_sec[i, j]
                                      >= self.pumpList[i].minPower * self.U_pumpDis[i, j])
                self.m.add_constraint(self.P_pumpDis[i, j] - self.P_pump_reg[i, j] - self.P_pump_sec[i, j]
                                      >= self.pumpList[i].minLFC * self.U_pumpDis[i, j])
                # get AGC_sec
                # pumpAGC_sec_up = self.m.min(self.pumpList[i].maxLFC - self.P_pumpDis[i, j],
                #                             10 * self.pumpList[i].rampUpLimit)
                # pumpAGC_sec_down = self.m.min(self.P_pumpDis[i, j] - self.pumpList[i].minLFC,
                #                               10 * self.genList[i].rampDownLimit)
                # self.m.add_constraint(self.P_pumpAGC_sec[i, j] == self.m.min(pumpAGC_sec_up, pumpAGC_sec_down))

                ## Tertiary constraints
                # 발전 중인 경우(Spinning)
                self.m.add_constraint(self.P_pumpDis[i, j] + self.P_pump_pri[i, j]
                                      + self.P_pump_reg[i, j] + self.P_pump_sec[i, j]
                                      + self.P_pump_terSpin[i, j]
                                      <= self.pumpList[i].AVAC * self.U_pumpDis[i, j]
                                      + self.pumpList[i].maxPump * self.U_pumpChg[i, j])
                self.m.add_constraint(self.P_pump_terSpin[i, j]
                                      <= self.pumpList[i].AVAC * self.U_pumpDis[i, j]
                                      + self.pumpList[i].maxPump * self.U_pumpChg[i, j])
                # self.m.add_constraint(self.P_pumpSpin[i, j]
                #                       == self.m.min(self.pumpList[i].AVAC - self.P_pumpDis[i, j],
                #                                     30 * self.pumpList[i].rampUpLimit))
                # 정지 상태(Nspinnning)
                self.m.add_constraint(self.P_pump_terNspin[i, j] <= self.pumpList[i].AVAC * (1 - self.U_pumpDis[i, j]))
                self.m.add_constraint(self.P_pump_terNspin[i, j] <= self.pumpList[i].AVAC * (1 - self.U_pumpChg[i, j]))

                ## Rapid(속응성) constraints
                # 정지 상태만 고려
                self.m.add_constraint(self.P_pump_rap[i, j] <= self.pumpList[i].AVAC * (1 - self.U_pumpDis[i, j]))
                self.m.add_constraint(self.P_pump_rap[i, j] <= self.pumpList[i].AVAC * (1 - self.U_pumpChg[i, j]))
                self.m.add_constraint(self.P_pump_terNspin[i, j] + self.P_pump_rap[i, j]
                                      <= self.pumpList[i].AVAC * (1 - self.U_pumpDis[i, j]))
                self.m.add_constraint(self.P_pump_terNspin[i, j] + self.P_pump_rap[i, j]
                                      <= self.pumpList[i].AVAC * (1 - self.U_pumpChg[i, j]))

            ## Non-spinning reserve
            # self.m.add_constraint(sum(self.P_pumpNspin[i, j] for i in range(self.nPump))
            #                      <= RESX * sum(self.pumpList[i].maxCapacity for i in range(self.nPump)))

    def constraint_reserve_req(self, revReqDict, cascading_flag):

        req_pri = revReqDict['REQ_pri']
        req_reg = revReqDict['REQ_reg']
        req_sec = revReqDict['REQ_sec']
        req_ter = revReqDict['REQ_ter']
        req_seq = revReqDict['REQ_seq']

        for j in range(self.nTimeslot):
            P_sum_pri = sum(self.P_gen_pri[i, j] for i in range(self.nGen)) \
                       + sum(self.P_pump_pri[i, j] for i in range(self.nPump))
            P_sum_reg = sum(self.P_gen_reg[i, j] for i in range(self.nGen)) \
                           + sum(self.P_pump_reg[i, j] for i in range(self.nPump))
            P_sum_sec = sum(self.P_gen_sec[i, j] for i in range(self.nGen)) \
                            + sum(self.P_pump_sec[i, j] for i in range(self.nPump))
            P_sum_ter = sum(self.P_gen_ter[i, j] for i in range(self.nGen)) \
                         + sum(self.P_pump_terSpin[i, j] for i in range(self.nPump)) \
                         + sum(self.P_pump_terNspin[i, j] for i in range(self.nPump))
            P_sum_rap = sum(self.P_pump_rap[i, j] for i in range(self.nPump))
                        # + sum(self.P_gen_rap[i, j] for i in range(self.nGen))

            if cascading_flag:
                self.m.add_constraint(P_sum_pri >= req_pri)
                self.m.add_constraint(P_sum_pri + P_sum_reg >= req_pri + req_reg)
                self.m.add_constraint(P_sum_pri + P_sum_reg + P_sum_sec >= req_pri + req_reg + req_sec)
                self.m.add_constraint(P_sum_pri + P_sum_reg + P_sum_sec + P_sum_ter
                                      >= req_pri + req_reg + req_sec + req_ter)
                self.m.add_constraint(P_sum_rap >= req_seq)
            else:
                self.m.add_constraint(P_sum_pri >= req_pri)
                self.m.add_constraint(P_sum_reg >= req_reg)
                self.m.add_constraint(P_sum_reg + P_sum_sec >= req_reg + req_sec)
                self.m.add_constraint(P_sum_reg + P_sum_sec + P_sum_ter >= req_reg + req_sec + req_ter)
                self.m.add_constraint(
                    P_sum_pri + P_sum_reg + P_sum_sec + P_sum_ter >= req_pri + req_reg + req_sec + req_ter)
                self.m.add_constraint(P_sum_rap >= req_seq)

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
                                        for i in range(self.nGen) for k in range(self.N_PIECE) for j in
                                        range(self.nTimeslot)) * self.UNIT_TIME
                             + sum(self.U_gen[i, j] * self.genList[i].a * self.UNIT_TIME * self.genList[i].fuelCost
                                   + self.SU_gen[i, j] * self.genList[i].startupCost
                                   + self.SD_gen[i, j] * self.genList[i].shutdownCost for i in range(self.nGen) for j in
                                   range(self.nTimeslot))
                             + sum(
            self.nuclearList[i].slopes[k] * self.P_nuclear[i, k, j] * self.nuclearList[i].fuelCost for i in
            range(self.nNuclear)
            for k in range(self.N_PIECE) for j in range(self.nTimeslot)) * self.UNIT_TIME
                             + sum(
            self.U_nuclear[i, j] * self.nuclearList[i].a * self.UNIT_TIME * self.nuclearList[i].fuelCost
            + self.SU_nuclear[i, j] * self.nuclearList[i].startupCost
            + self.SD_nuclear[i, j] * self.nuclearList[i].shutdownCost for i in range(self.nNuclear) for j in
            range(self.nTimeslot)))

    def solve(self, tol, timelimit):
        self.m.parameters.mip.tolerances.mipgap = tol
        self.m.parameters.timelimit = 60*timelimit
        self.m.print_information()
        new_sol = self.m.new_solution()
        # print(new_sol.find_unsatisfied_constraints(self.m))
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
        P_gen_pri = np.zeros((self.nGen, self.nTimeslot))
        P_gen_reg = np.zeros((self.nGen, self.nTimeslot))
        P_gen_sec = np.zeros((self.nGen, self.nTimeslot))
        P_gen_ter = np.zeros((self.nGen, self.nTimeslot))
        P_gen_rap = np.zeros((self.nGen, self.nTimeslot))

        P_pump_pri = np.zeros((self.nPump, self.nTimeslot))
        P_pump_reg = np.zeros((self.nPump, self.nTimeslot))
        P_pump_sec = np.zeros((self.nPump, self.nTimeslot))
        P_pump_terSpin = np.zeros((self.nPump, self.nTimeslot))
        P_pump_terNspin = np.zeros((self.nPump, self.nTimeslot))
        P_pump_rap = np.zeros((self.nPump, self.nTimeslot))

        for j in range(self.nTimeslot):
            for i in range(self.nGen):
                P_gen_pri[i, j] = sol.get_value(self.P_gen_pri[i, j])
                P_gen_reg[i, j] = sol.get_value(self.P_gen_reg[i, j])
                P_gen_sec[i, j] = sol.get_value(self.P_gen_sec[i, j])
                P_gen_ter[i, j] = sol.get_value(self.P_gen_ter[i, j])
                P_gen_rap[i, j] = sol.get_value(self.P_gen_rap[i, j])
            for i in range(self.nPump):
                P_pump_pri[i, j] = sol.get_value(self.P_pump_pri[i, j])
                P_pump_reg[i, j] = sol.get_value(self.P_pump_reg[i, j])
                P_pump_sec[i, j] = sol.get_value(self.P_pump_sec[i, j])
                P_pump_terSpin[i, j] = sol.get_value(self.P_pump_terSpin[i, j])
                P_pump_terNspin[i, j] = sol.get_value(self.P_pump_terNspin[i, j])
                P_pump_rap[i, j] = sol.get_value(self.P_pump_rap[i, j])

        reserveDict = {'RES_pri': [P_gen_pri, P_pump_pri],
                       'RES_reg': [P_gen_reg, P_pump_reg],
                       'RES_sec': [P_gen_sec, P_pump_sec],
                       'RES_ter': [P_gen_ter, P_pump_terSpin, P_pump_terNspin],
                       'RES_seq': [P_gen_rap, P_pump_rap]}

        return reserveDict

    def save_sol(self, date, P_sol, resSolDict, totalGen_Flag):
        [P_genSol, P_nuclearSol, P_pumpDisSol, P_pumpChgSol, P_essDisSol, P_essChgSol] = P_sol
        [P_gen_pri, P_pump_pri] = resSolDict['RES_pri']
        [P_gen_reg, P_pump_reg] = resSolDict['RES_reg']
        [P_gen_sec, P_pump_sec] = resSolDict['RES_sec']
        [P_gen_ter, P_pump_terSpin, P_pump_terNspin] = resSolDict['RES_ter']
        [P_gen_seq, P_pump_seq] = resSolDict['RES_seq']

        if totalGen_Flag:
            genName = ['Gen_sum', 'Gen_sum_1차', 'Gen_sum_주파수제어', 'Gen_sum_2차', 'Gen_sum_3차', 'Gen_sum_속응성']
            nuName = ['Nuclear_sum']
            genData = np.zeros((len(genName), len(self.load)))
            genData[0, :] = sum(P_genSol)
            genData[1, :] = sum(P_gen_pri)
            genData[2, :] = sum(P_gen_reg)
            genData[3, :] = sum(P_gen_sec)
            genData[4, :] = sum(P_gen_ter)
            genData[5, :] = sum(P_gen_seq)
            nuData = np.reshape(sum(P_nuclearSol), (1, -1))
        else:
            genName = [self.genList[i].name for i in range(self.nGen)] + \
                      [self.genList[i].name + '_1차' for i in range(self.nGen)] + \
                      [self.genList[i].name + '_주파수제어' for i in range(self.nGen)] + \
                      [self.genList[i].name + '_2차' for i in range(self.nGen)] + \
                      [self.genList[i].name + '_3차' for i in range(self.nGen)] + \
                      [self.genList[i].name + '_속응성' for i in range(self.nGen)]
            nuName = [self.nuclearList[i].name for i in range(self.nNuclear)]
            genData = np.vstack((P_genSol, P_gen_pri, P_gen_reg, P_gen_sec, P_gen_ter, P_gen_seq))
            nuData = P_nuclearSol

        pumpName = [self.pumpList[i].name for i in range(self.nPump)] + \
                   [self.pumpList[i].name + '_1차' for i in range(self.nPump)] + \
                   [self.pumpList[i].name + '_주파수제어' for i in range(self.nPump)] + \
                   [self.pumpList[i].name + '_2차' for i in range(self.nPump)] + \
                   [self.pumpList[i].name + '_3차(운전)' for i in range(self.nPump)] + \
                   [self.pumpList[i].name + '_3차(정지)' for i in range(self.nPump)] + \
                   [self.pumpList[i].name + '_속응성' for i in range(self.nPump)]
        pumpData = np.vstack((P_pumpDisSol - P_pumpChgSol, P_pump_pri, P_pump_reg, P_pump_sec,
                              P_pump_terSpin, P_pump_terNspin, P_pump_seq))

        df = pd.DataFrame(data=np.float32(np.vstack((self.load, genData, nuData, pumpData))),
                          index=['수요예측_'+str(date)]+genName + nuName + pumpName,
                          columns=[i for i in range(1, len(self.load)+1)])
        df.to_csv("./results_"+time.strftime("%m%d%H%M")+".csv", float_format='%.2f', encoding='utf-8-sig')
