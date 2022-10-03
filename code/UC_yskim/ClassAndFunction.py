### 발전기 객체 모델 / 단위: Power in MW, Ramp in MW/min
class ThermalGenerator:
    def __init__(self, name, busNumber, minMaxPower, rampUpDownLimit, costCoeff, numberOfPieces, minMaxLFC=[0,0], maxAGCUpDown=[0,0], minMaxGF=[0,0], maxSpinNspin=[0,0]):
        self.name = name
        self.busNumber = busNumber
        self.minPower = minMaxPower[0]
        self.maxPower = minMaxPower[1]
        self.rampUpLimit = rampUpDownLimit[0]
        self.rampDownLimit = rampUpDownLimit[1]
        self.a = costCoeff[0]
        self.startupCost = costCoeff[3]
        self.shutdownCost = costCoeff[4]
        self.minLFC = minMaxLFC[0]
        self.maxLFC = minMaxLFC[1]
        self.maxAGCUp = maxAGCUpDown[0]
        self.maxAGCDown = maxAGCUpDown[1]
        self.minGF = minMaxGF[0]
        self.maxGF = minMaxGF[1]
        self.maxSpin = maxSpinNspin[0]
        self.maxNspin = maxSpinNspin[1]

        slopesOfCostFunction = np.zeros(numberOfPieces)
        self.maxPowerPerPiece = self.maxPower / numberOfPieces
        for i in range(numberOfPieces):
            slopesOfCostFunction[i] = (costCoeff[0] + costCoeff[1] * (self.maxPowerPerPiece * (i + 1))
                                       + costCoeff[2] * (self.maxPowerPerPiece * (i + 1)) ** 2
                                       - costCoeff[0] - costCoeff[1] * self.maxPowerPerPiece * i
                                       - costCoeff[2] * (self.maxPowerPerPiece * i) ** 2) / self.maxPowerPerPiece
        self.slopes = slopesOfCostFunction

class PumpedStorage:
    def __init__(self, name, busNumber, minMaxPowerPump, rampUpDownLimit, initTermSOC, maxCapacity, efficiency, isFixedSpeed, fixedPumpPower, minMaxLFC=[0,0], maxAGCUpDown=[0,0], minMaxGF=[0,0], maxSpinNspin=[0,0]):
        self.name = name
        self.busNumber = busNumber
        self.minPower = minMaxPowerPump[0]
        self.maxPower = minMaxPowerPump[1]
        self.minPump = minMaxPowerPump[2]
        self.maxPump = minMaxPowerPump[3]
        self.rampUpLimit = rampUpDownLimit[0]
        self.rampDownLimit = rampUpDownLimit[1]
        self.initSOC = initTermSOC[0]
        self.termSOC = initTermSOC[1]
        self.maxCapacity = maxCapacity
        self.efficiency = efficiency
        self.isFixedSpeed = isFixedSpeed
        self.fixedPumpPower = fixedPumpPower
        self.minLFC = minMaxLFC[0]
        self.maxLFC = minMaxLFC[1]
        self.maxAGCUp = maxAGCUpDown[0]
        self.maxAGCDown = maxAGCUpDown[1]
        self.minGF = minMaxGF[0]
        self.maxGF = minMaxGF[1]
        self.maxSpin = maxSpinNspin[0]
        self.maxNspin = maxSpinNspin[1]

class EnergyStorage:
    def __init__(self, name, busNumber, minMaxPower, initTermSOC, maxCapacity, efficiency):
        self.name = name
        self.busNumber = busNumber
        self.minPower = minMaxPower[0]
        self.maxPower = minMaxPower[1]
        self.initSOC = initTermSOC[0]
        self.termSOC = initTermSOC[1]
        self.maxCapacity = maxCapacity
        self.efficiency = efficiency

class MILP:
    def __init__(self, NAME, generatorList, pumpList, essList, load, UNIT_TIME, N_PIECE, bus, branch, flow_limit):

        self.m = Model(name=NAME)

        self.generatorList = generatorList
        self.pumpList = pumpList
        self.essList = essList
        self.load = load
        self.UNIT_TIME = UNIT_TIME
        self.nGen = len(generatorList)
        self.nPump = len(pumpList)
        self.nEss = len(essList)
        self.nTimeslot = len(load)
        self.N_PIECE = N_PIECE
        self.bus = bus
        self.branch = branch
        self.nBus = np.shape(bus)[0]
        self.nBranch = np.shape(branch)[0]
        self.FLOW_LIMIT = flow_limit*np.ones(self.nBranch)
        self.B, _ = self.runDcPowerFlow(bus, branch)

        ##var_Gen
        for i in range(self.nGen):
            self.P_gen = self.m.continuous_var_cube(self.nGen, self.N_PIECE, self.nTimeslot, lb=0, ub=self.generatorList[i].maxPowerPerPiece)  # ub list로 수정해야
            self.P_genGF = self.m.continuous_var_matrix(self.nGen, self.nTimeslot, lb=self.generatorList[i].minGF, ub=self.generatorList[i].maxGF)
            self.P_genAGCup = self.m.continuous_var_matrix(self.nGen, self.nTimeslot, lb=0, ub=self.generatorList[i].maxAGCUp)
            self.P_genAGCdown = self.m.continuous_var_matrix(self.nGen, self.nTimeslot, lb=0, ub=self.generatorList[i].maxAGCDown)
            self.P_genSpin = self.m.continuous_var_matrix(self.nGen, self.nTimeslot, lb=0, ub=self.generatorList[i].maxSpin)
            self.P_genNspin = self.m.continuous_var_matrix(self.nGen, self.nTimeslot, lb=0, ub=self.generatorList[i].maxNspin)
        self.U_gen = self.m.binary_var_matrix(self.nGen, self.nTimeslot)
        self.SU_gen = self.m.binary_var_matrix(self.nGen, self.nTimeslot)
        self.SD_gen = self.m.binary_var_matrix(self.nGen, self.nTimeslot)
        self.U_gen_agc = self.m.binary_var_matrix(self.nGen, self.nTimeslot)
        ##var_Pump
        for i in range(self.nPump):
            self.P_pumpChg = self.m.continuous_var_matrix(self.nPump, self.nTimeslot, lb=0, ub=self.pumpList[i].maxPower)
            self.P_pumpDis = self.m.continuous_var_matrix(self.nPump, self.nTimeslot, lb=0, ub=self.pumpList[i].maxPump)
            self.P_pumpGF = self.m.continuous_var_matrix(self.nPump, self.nTimeslot, lb=self.pumpList[i].minGF, ub=self.pumpList[i].maxGF)
            self.P_pumpAGCup = self.m.continuous_var_matrix(self.nPump, self.nTimeslot, lb=0, ub=self.pumpList[i].maxAGCUp)
            self.P_pumpAGCdown = self.m.continuous_var_matrix(self.nPump, self.nTimeslot, lb=0, ub=self.pumpList[i].maxAGCDown)
            self.P_pumpSpin = self.m.continuous_var_matrix(self.nPump, self.nTimeslot, lb=0, ub=self.pumpList[i].maxSpin)
            self.P_pumpNspin = self.m.continuous_var_matrix(self.nPump, self.nTimeslot, lb=0, ub=self.pumpList[i].maxNspin)
        self.U_pumpChg = self.m.binary_var_matrix(self.nPump, self.nTimeslot)
        self.U_pumpDis = self.m.binary_var_matrix(self.nPump, self.nTimeslot)
        ##var_ESS
        for i in range(self.nEss):
            self.P_essChg = self.m.continuous_var_matrix(self.nEss, self.nTimeslot, lb=0, ub=self.essList[i].maxPower)
            self.P_essDis = self.m.continuous_var_matrix(self.nEss, self.nTimeslot, lb=0, ub=self.essList[i].maxPower)
        self.U_essChg = self.m.binary_var_matrix(self.nEss, self.nTimeslot)
        self.U_essDis = self.m.binary_var_matrix(self.nEss, self.nTimeslot)

    def constraint_minMax(self):
        for j in range(self.nTimeslot):
            for i in range(self.nGen):
                self.m.add_constraint(sum(self.P_gen[i, k, j] for k in range(self.N_PIECE)) >= self.U_gen[i, j] * self.generatorList[i].minPower)
                self.m.add_constraint(sum(self.P_gen[i, k, j] for k in range(self.N_PIECE)) <= self.U_gen[i, j] * self.generatorList[i].maxPower)
                self.m.add_constraint(self.SU_gen[i, j] + self.SD_gen[i, j] <= 1)
                self.m.add_constraint(sum(self.P_gen[i, k, j] for k in range(self.N_PIECE))
                                 + sum(self.P_pumpDis[l, j] for l in range(self.nPump))
                                 - sum(self.P_pumpChg[l, j] for l in range(self.nPump))
                                 + sum(self.P_essDis[l, j] for l in range(self.nEss))
                                 - sum(self.P_essChg[l, j] for l in range(self.nEss)) == self.load[j])
            for i in range(self.nPump):
                self.m.add_constraint(self.P_pumpDis[i, j] <= self.U_pumpDis[i, j] * self.pumpList[i].maxPower)
                self.m.add_constraint(self.P_pumpDis[i, j] >= self.U_pumpDis[i, j] * self.pumpList[i].minPower)
                self.m.add_constraint(self.P_pumpChg[i, j]
                                 <= self.U_pumpChg[i, j] *
                                 (self.pumpList[i].maxPump - 0.9999 * self.pumpList[i].isFixedSpeed * (
                                             self.pumpList[i].maxPump - self.pumpList[i].fixedPumpPower)))
                self.m.add_constraint(self.P_pumpChg[i, j]
                                 >= self.U_pumpChg[i, j] *
                                 (self.pumpList[i].minPump + 0.9999 * self.pumpList[i].isFixedSpeed * (
                                             self.pumpList[i].fixedPumpPower - self.pumpList[i].minPump)))
                self.m.add_constraint(self.U_pumpDis[i, j] + self.U_pumpChg[i, j] <= 1)
            for i in range(self.nEss):
                self.m.add_constraint(self.P_essDis[i, j] <= self.U_essDis[i, j] * self.essList[i].maxPower)
                self.m.add_constraint(self.P_essChg[i, j] <= self.U_essChg[i, j] * self.essList[i].maxPower)
                self.m.add_constraint(self.U_essDis[i, j] + self.U_essChg[i, j] <= 1)

    def constraint_SOC(self, minMaxSoCPump, initTermSoCPump, minMaxSoCEss, initTermSoCEss):
        for j in range(self.nTimeslot):
            for i in range(self.nPump):
                self.m.add_constraint(self.pumpList[i].initSOC
                                 - sum(self.P_pumpDis[i, k] * self.UNIT_TIME / self.pumpList[i].maxCapacity  # 양수는 발전시 효율 곱하지 않음
                                       for k in range(j + 1))
                                 + sum(self.P_pumpChg[i, k] * self.UNIT_TIME * self.pumpList[i].efficiency / self.pumpList[i].maxCapacity
                                       for k in range(j + 1)) <= minMaxSoCPump[1])
                self.m.add_constraint(self.pumpList[i].initSOC
                                 - sum(self.P_pumpDis[i, k] * self.UNIT_TIME / self.pumpList[i].maxCapacity
                                       for k in range(j + 1))
                                 + sum(self.P_pumpChg[i, k] * self.UNIT_TIME * self.pumpList[i].efficiency / self.pumpList[i].maxCapacity
                                       for k in range(j + 1)) >= minMaxSoCPump[0])
            for i in range(self.nEss):
                self.m.add_constraint(self.essList[i].initSOC
                                 - sum(self.P_essDis[i, k] * self.UNIT_TIME / self.essList[i].efficiency / self.essList[i].maxCapacity
                                       for k in range(j + 1))
                                 + sum(self.P_essChg[i, k] * self.UNIT_TIME * self.essList[i].efficiency / self.essList[i].maxCapacity
                                       for k in range(j + 1)) <= minMaxSoCEss[1])
                self.m.add_constraint(self.essList[i].initSOC
                                 - sum(self.P_essDis[i, k] * self.UNIT_TIME / self.essList[i].efficiency / self.essList[i].maxCapacity
                                       for k in range(j + 1))
                                 + sum(self.P_essChg[i, k] * self.UNIT_TIME * self.essList[i].efficiency / self.essList[i].maxCapacity
                                       for k in range(j + 1)) >= minMaxSoCEss[0])
        for i in range(self.nPump):
            self.m.add_constraint(self.pumpList[i].initSOC
                             - sum(self.P_pumpDis[i, k] * self.UNIT_TIME / self.pumpList[i].maxCapacity for k in range(self.nTimeslot))
                             + sum(self.P_pumpChg[i, k] * self.UNIT_TIME * self.pumpList[i].efficiency / self.pumpList[i].maxCapacity for k in range(self.nTimeslot))
                             >= self.pumpList[i].termSOC)
        for i in range(self.nEss):
            self.m.add_constraint(self.essList[i].initSOC
                             - sum(self.P_essDis[i, k] * self.UNIT_TIME / self.essList[i].efficiency / self.essList[i].maxCapacity for k in range(self.nTimeslot))
                             + sum(self.P_essChg[i, k] * self.UNIT_TIME * self.essList[i].efficiency / self.essList[i].maxCapacity for k in range(self.nTimeslot))
                             == self.essList[i].termSOC)

    def constraint_rampUpDown(self):
        for j in range(self.nTimeslot - 1):
            # 발전기 rampup down 제약(reserve included)
            for i in range(self.nGen):
                self.m.add_constraint(sum(self.P_gen[i, k, j + 1] for k in range(self.N_PIECE)) - sum(self.P_gen[i, k, j] for k in range(self.N_PIECE))
                                      + self.P_genGF[i, j] + self.P_genAGCup[i, j] + self.P_genSpin[i, j]
                                      <= self.generatorList[i].rampUpLimit * 60 * self.UNIT_TIME)
                self.m.add_constraint(sum(self.P_gen[i, k, j + 1] for k in range(self.N_PIECE)) - sum(self.P_gen[i, k, j] for k in range(self.N_PIECE))
                                      - self.P_genGF[i, j] - self.P_genAGCdown[i, j]
                                      >= -self.generatorList[i].rampUpLimit * 60 * self.UNIT_TIME)
                # 발전기 ONOFF 제약
                self.m.add_constraint(self.U_gen[i, j + 1] - self.U_gen[i, j] == self.SU_gen[i, j + 1] - self.SD_gen[i, j + 1])
            # 양수 rampup down 제약(reserve included)
            for i in range(self.nPump):
                self.m.add_constraint(self.P_pumpDis[i, j + 1] - self.P_pumpDis[i, j] + self.P_pumpGF[i, j] + self.P_pumpAGCup[i, j] + self.P_pumpSpin[i, j]
                                      <= self.pumpList[i].rampUpLimit * 60 * self.UNIT_TIME)
                self.m.add_constraint(self.P_pumpDis[i, j + 1] - self.P_pumpDis[i, j] - self.P_pumpGF[i, j] - self.P_pumpAGCdown[i, j]
                                      >= -self.pumpList[i].rampDownLimit * 60 * self.UNIT_TIME)
                #self.m.add_constraint( self.P_pumpChg[i, j + 1] - self.P_pumpChg[i, j] <= self.pumpList[i].rampUpLimit*60*self.UNIT_TIME )
                #self.m.add_constraint( self.P_pumpChg[i, j + 1] - self.P_pumpChg[i, j] >= -self.pumpList[i]rampDownLimit*60*self.UNIT_TIME )

    def constraint_gen_initState(self):
        for i in range(self.nGen):
            self.m.add_constraint(self.U_gen[i, 0] == self.SU_gen[i, 0])  # 시작할때부터 발전기 켜져있으면 startup 비용 추가

    def constraint_reserve(self, AGC_cascading_flag=True, RESX=0.5):
        ## Gen reserve
        for i in range(self.nGen):
            for j in range(self.nTimeslot):
                ## GF constraints
                self.m.add_constraint(sum(self.P_gen[i, k, j] for k in range(self.N_PIECE)) + self.P_genGF[i, j] <= self.generatorList[i].maxPower)
                self.m.add_constraint(sum(self.P_gen[i, k, j] for k in range(self.N_PIECE)) - self.P_genGF[i, j] >= self.generatorList[i].minPower)
                if AGC_cascading_flag:
                    ## AGC constraints
                    self.m.add_constraint(sum(self.P_gen[i, k, j] for k in range(self.N_PIECE)) + self.P_genGF[i, j] + self.P_genAGCup[i, j]
                                          <= self.generatorList[i].maxPower)
                    self.m.add_constraint(sum(self.P_gen[i, k, j] for k in range(self.N_PIECE)) + self.P_genAGCup[i, j]
                                          <= generatorList[i].maxLFC * self.U_gen_agc[i, j] + self.generatorList[i].maxPower * (1 - self.U_gen_agc[i, j]))
                    self.m.add_constraint(self.P_genAGCup[i, j] + self.P_genGF[i, j]
                                          <= generatorList[i].maxAGCUp * self.U_gen_agc[i, j] + generatorList[i].maxGF * (1 - self.U_gen_agc[i, j]))
                    self.m.add_constraint(sum(self.P_gen[i, k, j] for k in range(self.N_PIECE)) - self.P_genGF[i, j] - self.P_genAGCdown[i, j]
                                          >= self.generatorList[i].minPower)
                    self.m.add_constraint(sum(self.P_gen[i, k, j] for k in range(self.N_PIECE)) - self.P_genAGCdown[i, j]
                                          >= self.generatorList[i].minLFC * self.U_gen_agc[i, j] + self.generatorList[i].minPower * (1 - self.U_gen_agc[i, j]))
                    self.m.add_constraint(self.P_genAGCdown[i, j] + self.P_genGF[i, j]
                                          <= self.generatorList[i].maxAGCDown * self.U_gen_agc[i, j] + self.generatorList[i].maxGF * (1 - self.U_gen_agc[i, j]))
                    ## Spinning constraints
                    self.m.add_constraint(sum(self.P_gen[i, k, j] for k in range(self.N_PIECE)) + self.P_genGF[i, j]
                                          + self.P_genAGCup[i, j] + self.P_genSpin[i, j] <= self.generatorList[i].maxPower)
                else:
                    ## AGC constraint
                    self.m.add_constraint(self.P_genAGCup[i, j] <= generatorList[i].maxAGCUp * self.U_gen_agc[i, j])
                    self.m.add_constraint(self.P_genAGCdown[i, j] <= generatorList[i].maxAGCDown * self.U_gen_agc[i, j])
                    ## Spinning constraints
                    self.m.add_constraint(sum(self.P_gen[i, k, j] for k in range(self.N_PIECE))
                                          + self.P_genGF[i, j] + self.P_genSpin[i, j] <= self.generatorList[i].maxPower)
                ## Nonspinning constraints
                self.m.add_constraint(self.P_genNspin[i, j] <= self.generatorList[i].maxNspin * (1 - self.U_gen[i, j]))
        ## Pump reserve
        for j in range(self.nTimeslot):
            for i in range(self.nGen):
                ## GF constraints
                self.m.add_constraint(self.P_pumpDis[i, j] + self.P_pumpGF[i, j] <= self.pumpList[i].maxPump * self.U_pumpDis[i, j])
                ## AGC constraints
                self.m.add_constraint(self.P_pumpDis[i, j] + self.P_pumpGF[i, j] + self.P_pumpAGCup[i, j] <= pumpList[i].maxLFC * self.U_pumpDis[i, j])
                self.m.add_constraint(self.P_pumpDis[i, j] - self.P_pumpAGCdown[i, j] >= pumpList[i].minLFC * self.U_pumpDis[i, j])
                ## Spinning constraints
                self.m.add_constraint(self.P_pumpDis[i, j] + self.P_pumpGF[i, j] + self.P_pumpAGCup[i, j] + self.P_pumpSpin[i, j]
                                      <= self.pumpList[i].maxPump * self.U_pumpDis[i, j] + pumpList[i].maxPump * self.U_pumpChg[i, j])
                self.m.add_constraint(self.P_pumpSpin[i, j] <= self.pumpList[i].maxSpin * self.U_pumpDis[i, j] + self.pumpList[i].maxPump * self.U_pumpChg[i, j])
                ## Nonspinning constraints
                self.m.add_constraint(self.P_pumpNspin[i, j] <= self.pumpList[i].maxNspin * (1 - self.U_pumpDis[i, j]))
                self.m.add_constraint(self.P_pumpNspin[i, j] <= self.pumpList[i].maxNspin * (1 - self.U_pumpChg[i, j]))
            ## Non-spinning reserve
            self.m.add_constraint(sum(self.P_pumpNspin[i, j] for i in range(self.nPump))
                                  <= RESX*sum(self.pumpList[i].maxCapacity for i in range(self.nPump)))

    def constraint_reserve_req(self, method_flag, REQ):
        [req_GF, req_AGC, req_Spin, req_Nspin] = REQ
        for j in range(self.nTimeslot):
            P_sum_GF = sum(self.P_genGF[i, j] for i in range(self.nGen)) + sum(self.P_pumpGF[i, j] for i in range(self.nPump))
            P_sum_AGCup = sum(self.P_genAGCup[i, j] for i in range(self.nGen)) + sum(self.P_pumpAGCup[i, j] for i in range(self.nPump))
            P_sum_AGCdown = sum(self.P_genAGCdown[i, j] for i in range(self.nGen)) + sum(self.P_pumpAGCdown[i, j] for i in range(self.nPump))
            P_sum_Spin = sum(self.P_genSpin[i, j] for i in range(self.nGen)) + sum(self.P_pumpSpin[i, j] for i in range(self.nPump))
            P_sum_Nspin = sum(self.P_genNspin[i, j] for i in range(self.nGen)) + sum(self.P_pumpNspin[i, j] for i in range(self.nPump))
            if method_flag:
                self.m.add_constraint(P_sum_GF >= req_GF)
                self.m.add_constraint(P_sum_GF + P_sum_AGCup >= req_GF + req_AGC)
                self.m.add_constraint(P_sum_GF + P_sum_AGCup + P_sum_Spin >= req_GF + req_AGC + req_Spin)
                self.m.add_constraint(P_sum_GF + P_sum_AGCup + P_sum_Spin + P_sum_Nspin >= req_GF + req_AGC + req_Spin + req_Nspin)
                self.m.add_constraint(P_sum_GF + P_sum_AGCdown >= req_GF + req_AGC)
            else:
                self.m.add_constraint(P_sum_GF >= req_GF)
                self.m.add_constraint(P_sum_GF + P_sum_Spin >= req_GF + req_Spin)
                self.m.add_constraint(P_sum_GF + P_sum_Spin + P_sum_Nspin >= req_GF + req_Spin + req_Nspin)
                self.m.add_constraint(P_sum_AGCup >= req_AGC)
                self.m.add_constraint(P_sum_AGCdown >= req_AGC)

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

        theta = B * busNetPower  # swing bus를 제외한 theta 계산
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
                    if self.generatorList[l].busNumber == k + 2:  # slack 모선을 제외했으므로 index 2가 0번째 row가 됨
                        busNetPower_var[i][k] = busNetPower_var[i][k] + sum(self.P_gen[l, m, t] for m in range(self.N_PIECE))
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
                self.m.add_constraint(-self.branch[k, 2, None][0] * (theta_var[i][int(self.branch[k, 0, None] - 1), 0] - theta_var[i][int(self.branch[k, 1, None] - 1), 0]) <= self.FLOW_LIMIT[k])
                self.m.add_constraint(-self.branch[k, 2, None][0] * (theta_var[i][int(self.branch[k, 0, None] - 1), 0] - theta_var[i][int(self.branch[k, 1, None] - 1), 0]) >= -self.FLOW_LIMIT[k])

    def check_flow_limit(self, violationList, sol):
        [P_genSol, P_pumpDisSol, P_pumpChgSol, P_essDisSol, P_essChgSol] = sol
        branch = np.loadtxt('branch.txt', delimiter = ',', skiprows = 1, dtype = float)
        cnt_violation = 0
        for i in range(self.nTimeslot):
            bus = np.loadtxt('bus.txt', delimiter = ',', skiprows = 1, dtype = float)
            for k in range(self.nBus - 1):
                for l in range(self.nGen):
                    if self.generatorList[l].busNumber == k + 2:  # slack 모선을 제외했으므로 index 2가 0번째 row가 됨
                        bus[k + 1, 1, None] = bus[k + 1, 1, None] + P_genSol[l, i]
                for l in range(self.nPump):
                    if self.pumpList[l].busNumber == k + 2:
                        bus[k + 1, 1, None] = bus[k + 1, 1, None] + P_pumpDisSol[l, i] - P_pumpChgSol[l, i]
                for l in range(self.nEss):
                    if self.essList[l].busNumber == k + 2:
                        bus[k + 1, 1, None] = bus[k + 1, 1, None] + P_essDisSol[l, i] - P_essChgSol[l, i]

            # 총부하량을 모선별 분배 할당 *** 향후 개선 필요
            for n in range(self.nBus - 1):
                bus[n+1, 2, None] = bus[n+1, 2, None] + self.load[i] / (self.nBus - 1)

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
        self.m.set_objective("min", sum(self.generatorList[i].slopes[k] * self.P_gen[i, k, j] for i in range(self.nGen) for k in range(self.N_PIECE)
                            for j in range(self.nTimeslot)) * self.UNIT_TIME + sum(self.U_gen[i, j] * self.generatorList[i].a * self.UNIT_TIME + self.SU_gen[i, j] * self.generatorList[i].startupCost
                            + self.SD_gen[i, j] * self.generatorList[i].shutdownCost for i in range(self.nGen) for j in range(self.nTimeslot)))

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
            for j in range(self.nPump):
                P_pumpDisSol[j, i] = sol.get_value(self.P_pumpDis[j, i])
                P_pumpChgSol[j, i] = sol.get_value(self.P_pumpChg[j, i])
                U_pumpDisSol[j, i] = sol.get_value(self.U_pumpDis[j, i])
                U_pumpChgSol[j, i] = sol.get_value(self.U_pumpChg[j, i])
                socPump[j, i] = (self.pumpList[j].initSOC
                                 - sum(P_pumpDisSol[j, k] * self.UNIT_TIME / self.pumpList[j].efficiency / self.pumpList[j].maxCapacity for k in range(i + 1))
                                 + sum(P_pumpChgSol[j, k] * self.UNIT_TIME * self.pumpList[j].efficiency / self.pumpList[j].maxCapacity for k in range(i + 1)))
            for j in range(self.nEss):
                P_essDisSol[j, i] = sol.get_value(self.P_essDis[j, i])
                P_essChgSol[j, i] = sol.get_value(self.P_essChg[j, i])
                U_essDisSol[j, i] = sol.get_value(self.U_essDis[j, i])
                U_essChgSol[j, i] = sol.get_value(self.U_essChg[j, i])
                socEss[j, i] = (self.essList[j].initSOC
                                - sum(P_essDisSol[j, k] * self.UNIT_TIME / self.essList[j].efficiency / self.essList[j].maxCapacity for k in range(i + 1))
                                + sum(P_essChgSol[j, k] * self.UNIT_TIME * self.essList[j].efficiency / self.essList[j].maxCapacity for k in range(i + 1)))

        return [P_genSol, P_pumpDisSol, P_pumpChgSol, P_essDisSol, P_essChgSol],\
               [U_genSol, SU_genSol, SD_genSol, U_pumpDisSol, U_pumpChgSol, U_essDisSol, U_essChgSol],\
               [socPump, socEss]

    def make_plot(self, P_sol, U_sol, SoC_sol, save_flag=False):
        [P_genSol, P_pumpDisSol, P_pumpChgSol, P_essDisSol, P_essChgSol] = P_sol
        [U_genSol, SU_genSol, SD_genSol, U_pumpDisSol, U_pumpChgSol, U_essDisSol, U_essChgSol] = U_sol
        [socPump, socESS] = SoC_sol

        plt.rcParams["figure.figsize"] = (6, 6)
        plt.figure(1)
        plt.subplot(211)
        plt.plot(np.arange(self.nTimeslot), self.load, label='Load')
        for i in range(self.nGen):
            plt.plot(np.arange(self.nTimeslot), P_genSol[i, :], label='P_gen_' + str(i))
        for i in range(self.nPump):
            plt.plot(np.arange(self.nTimeslot), P_pumpDisSol[i, :], label='P_pumpDis_' + str(i))
        for i in range(self.nPump):
            plt.plot(np.arange(self.nTimeslot), -P_pumpChgSol[i, :], label='P_pumpChg_' + str(i))
        for i in range(self.nEss):
            plt.plot(np.arange(self.nTimeslot), P_essDisSol[i, :], label='P_essDis_' + str(i))
        for i in range(self.nEss):
            plt.plot(np.arange(self.nTimeslot), -P_essChgSol[i, :], label='P_essChg_' + str(i))
        plt.legend(loc='upper right', ncol=3, fontsize='x-small')
        plt.grid()
        plt.xticks(np.arange(0, self.nTimeslot, 1/self.UNIT_TIME), fontsize=8)
        plt.ylabel('Power (MW)')
        plt.xlim([0, self.nTimeslot-1])
        plt.subplot(212)
        n = 0
        plt.plot(np.arange(self.nTimeslot), socPump[n, :], label='SoC_pump' + str(n))
        plt.plot(np.arange(self.nTimeslot), socESS[n, :], label='SoC_ESS' + str(n))
        plt.legend(loc='upper right', ncol=2, fontsize='x-small')
        plt.grid()
        plt.xticks(np.arange(0, self.nTimeslot, 1 / self.UNIT_TIME), fontsize=8)
        plt.ylim([0, 1])
        plt.ylabel('SoC')
        plt.xlim([0, self.nTimeslot-1])
        if save_flag:
            plt.savefig('./test_total.png', dpi=300, bbox_inches='tight')
        #plt.show()

        plt.rcParams["figure.figsize"] = (6, 6)
        plt.figure(2)
        plt.subplot(411)
        for i in range(self.nGen):
            plt.plot(np.arange(self.nTimeslot), U_genSol[i, :], label='U_gen_' + str(i))
        plt.legend(loc='upper right', ncol=1, fontsize='x-small')
        plt.grid()
        plt.xticks(np.arange(0, self.nTimeslot, 4 / self.UNIT_TIME), fontsize=8)
        plt.ylim([-0.1, 1.1])
        plt.xlim([0, self.nTimeslot - 1])
        plt.ylabel('Binary')
        plt.subplot(412)
        for i in range(self.nGen):
            plt.plot(np.arange(self.nTimeslot), SU_genSol[i, :], label='SU_' + str(i))
            plt.plot(np.arange(self.nTimeslot), SD_genSol[i, :], label='SD_' + str(i))
        plt.legend(loc='upper right', ncol=2, fontsize='x-small')
        plt.grid()
        plt.xticks(np.arange(0, self.nTimeslot, 4 / self.UNIT_TIME), fontsize=8)
        plt.ylim([-0.1, 1.1])
        plt.xlim([0, self.nTimeslot - 1])
        plt.ylabel('Binary')
        plt.subplot(413)
        for i in range(self.nPump):
            plt.plot(np.arange(self.nTimeslot), U_pumpDisSol[i, :], label='U_pumpDis_' + str(i))
            plt.plot(np.arange(self.nTimeslot), U_pumpChgSol[i, :], label='U_pumpChg_' + str(i))
        plt.legend(loc='upper right', ncol=2, fontsize='x-small')
        plt.grid()
        plt.xticks(np.arange(0, self.nTimeslot, 4 / self.UNIT_TIME), fontsize=8)
        plt.ylim([-0.1, 1.1])
        plt.xlim([0, self.nTimeslot - 1])
        plt.ylabel('Binary')
        plt.subplot(414)
        for i in range(self.nEss):
            plt.plot(np.arange(self.nTimeslot), U_essDisSol[i, :], label='U_essDis_' + str(i))
            plt.plot(np.arange(self.nTimeslot), U_essChgSol[i, :], label='U_essChg_' + str(i))
        plt.legend(loc='upper right', ncol=2, fontsize='x-small')
        plt.grid()
        plt.ylim([-0.1, 1.1])
        plt.xlim([0, self.nTimeslot - 1])
        plt.xticks(np.arange(0, self.nTimeslot, 4 / self.UNIT_TIME), fontsize=8)
        plt.ylabel('Binary')
        if save_flag:
            plt.savefig('./test_binary.png', dpi=300, bbox_inches='tight')
        plt.show()

def make_model_list(N_PIECE, SOC, generatornamelist, pumpnamelist, essnamelist):
    [SOC_PUMP_INIT, SOC_PUMP_TERM, SOC_ESS_INIT, SOC_ESS_TERM] = SOC
    rsc_gen_input = pd.read_excel("발전기자료_2018년8월RSC자료(최종).xlsx", sheet_name='최종', header=3)
    tech_gen_Input1 = pd.read_excel("발전기자료_기술적특성자료.xlsx", sheet_name='1.기력,내연', header=4)
    tech_gen_Input2 = pd.read_excel("발전기자료_기술적특성자료.xlsx", sheet_name='2.복합_CC', header=4)
    tech_gen_Input3 = pd.read_excel("발전기자료_기술적특성자료.xlsx", sheet_name='2.복합_GT', header=4)
    tech_gen_Input4 = pd.read_excel("발전기자료_기술적특성자료.xlsx", sheet_name='3.원자력', header=4)
    tech_gen_Input5 = pd.read_excel("발전기자료_기술적특성자료.xlsx", sheet_name='4.수력,양수', header=4)

    tech_gen_list = [tech_gen_Input1, tech_gen_Input2, tech_gen_Input3, tech_gen_Input4]

    gen_name1 = tech_gen_Input1['발전기명'].unique()
    gen_name2 = tech_gen_Input2['발전기명'].unique()
    gen_name3 = tech_gen_Input3['발전기명'].unique()
    gen_name4 = tech_gen_Input4['발전기명'].unique()

    gen_name_list = [gen_name1, gen_name2, gen_name3, gen_name4]

    generatorList = []
    pumpList = []
    essList = []

    # make generatorList
    for i in range(len(generatornamelist)):
        rsc_generator = rsc_gen_input[rsc_gen_input['발전기명'] == generatornamelist[i]].iloc[0]
        for j in range(4):
            if generatornamelist[i] in gen_name_list[j]:
                tech_generator = tech_gen_list[j][tech_gen_list[j]['발전기명'] == generatornamelist[i]].iloc[0]
                name = tech_generator['발전기명']
                busNumber = 2  # 임의값
                if j in [0, 3]:
                    minMaxPower = [tech_generator['최소발전용량'], tech_generator['최대발전용량']]
                else:
                    minMaxPower = [tech_generator['최소발전용량(MW)'], tech_generator['최대발전용량(MW)']]
                rampUpDownLimit = [tech_generator['출력증가율'], tech_generator['출력감소율']]
                maxAGCUpDown = [minMaxPower[1]*0.5, minMaxPower[1]*0.5 ]  # 임의값
                if j in [0, 3]:
                    minMaxLFC = [tech_generator['AGC 하한'], tech_generator['AGC 상한']]
                    minMaxGF = [tech_generator['GF 하한'], tech_generator['GF 상한']]
                else:
                    minMaxLFC = [tech_generator['AGC 하한(MW)'], tech_generator['AGC 상한(MW)']]
                    minMaxGF = [tech_generator['GF 하한(MW)'], tech_generator['GF 상한(MW)']]
                maxSpinNspin = [minMaxPower[1]*0.5, minMaxPower[1]*0.5]
                costCoeff = [rsc_generator['상수'], rsc_generator['1차 계수'], rsc_generator['2차 계수'],
                             rsc_generator['HOT 기동\n비용(천원)'], rsc_generator['COLD\n기동비용']]

        print("ThermalGenerator({name}, busNumber={busNumber}, minMaxPower={minMaxPower}, rampUpDownLimit={rampUpDownLimit}, costCoeff={costCoeff},"
              " N_PIECE={N_PIECE}), minMaxLFC={minMaxLFC}, maxAGCUpDown={maxAGCUpDown}, minMaxGF={minMaxGF}, maxSpinNspin={maxSpinNspin})"
            .format(name=name,busNumber=busNumber, minMaxPower=minMaxPower, rampUpDownLimit=rampUpDownLimit, costCoeff=costCoeff, N_PIECE=N_PIECE, minMaxLFC=minMaxLFC, maxAGCUpDown=maxAGCUpDown, minMaxGF=minMaxGF, maxSpinNspin=maxSpinNspin))
        generatorList.append(ThermalGenerator(name, busNumber, minMaxPower, rampUpDownLimit, costCoeff, N_PIECE, minMaxLFC, maxAGCUpDown, minMaxGF, maxSpinNspin))  # busnumber,maxSpinNspin 임의값

    # make pumpList
    for i in range(len(pumpnamelist)):
        rsc_pump = rsc_gen_input[rsc_gen_input['발전기명'] == pumpnamelist[i]].iloc[0]
        tech_pump = tech_gen_Input5[tech_gen_Input5['발전소명'] == pumpnamelist[i]].iloc[0]

        name = tech_pump['발전소명']
        busNumber = 3  # 임의값
        minMaxPowerPump = [tech_pump['최소발전용량'], tech_pump['최대발전용량'], 0, tech_pump['최대발전용량']]  # minMaxPump 임의값
        rampUpDownLimit = [tech_pump['출력증가율'], tech_pump['출력감소율']]
        # [rsc_pump['상수'], rsc_pump['1차 계수'], rsc_pump['2차 계수'], rsc_pump['HOT 기동\n비용(천원)'], rsc_pump['COLD\n기동비용']] 양수발전은 costCoeff 없다?
        minMaxLFC = [tech_pump['AGC 하한'], tech_pump['AGC 상한']]
        maxAGCUpDown = [tech_pump['AGC 하한'], tech_pump['AGC 상한']]  # 임의값
        minMaxGF = [tech_pump['GF 하한'], tech_pump['GF 상한']]
        maxSpinNspin = [minMaxPowerPump[1]*0.5, minMaxPowerPump[1]*0.5]  # 임의값
        initTermSOC = [SOC_PUMP_INIT, SOC_PUMP_TERM]
        maxCapacity = tech_pump['설비용량']
        efficiency = 0.8  # 임의값
        isFixedSpeed = 1
        fixedPumpPower = 1

        print("PumpStorage({name}, busNumber={busNumber}, minMaxPowerPump={minMaxPowerPump}, rampUpDownLimit={rampUpDownLimit}, initTermSOC={initTermSOC},"
              " maxCapacity={maxCapacity}, efficiency={efficiency}, isFixedSpeed={isFixedSpeed}, fixedPumpPower={fixedPumpPower}, minMaxLFC={minMaxLFC}, maxAGCUpDown={maxAGCUpDown}, minMaxGF={minMaxGF}, maxSpinNspin={maxSpinNspin})"
            .format(name=name, busNumber=busNumber, minMaxPowerPump=minMaxPowerPump, rampUpDownLimit=rampUpDownLimit,
                initTermSOC=initTermSOC, maxCapacity=maxCapacity, efficiency=efficiency, isFixedSpeed=isFixedSpeed, fixedPumpPower=fixedPumpPower,
                minMaxLFC=minMaxLFC, maxAGCUpDown=maxAGCUpDown, minMaxGF=minMaxGF, maxSpinNspin=maxSpinNspin))
        pumpList.append(PumpedStorage(name, busNumber, minMaxPowerPump, rampUpDownLimit, initTermSOC, maxCapacity, efficiency, isFixedSpeed, fixedPumpPower, minMaxLFC, maxAGCUpDown, minMaxGF, maxSpinNspin))
    # make ESS
    essList.append(EnergyStorage('한전1', 4, [0, 50], [SOC_ESS_INIT, SOC_ESS_TERM], 70, 0.9))

    return generatorList, pumpList, essList