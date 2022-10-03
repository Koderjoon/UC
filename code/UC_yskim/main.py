import numpy as np
import matplotlib.pyplot as plt
from docplex.mp.model import Model
from gen_class import *
### Load data
load = [3, 3, 8, 3, 3, 3, 3, 3, 3, 3, 4, 4, 5, 6, 6, 6, 5, 4, 4, 5, 6, 4, 2, 0]
UNIT_TIME = 1 # Timeslot마다 에너지MWh를 구하기 위해 출력MW에 곱해져야하는 시간 단위, 1/4 = 15 min
# SOC_ESS_MIN = 0.3
# SOC_ESS_MAX = 0.8
# SOC_ESS_INIT = 0.5
# SOC_ESS_TERM = 0.5
N_PIECE = 3 # cost function을 선형화할 구간 수
costCoeff_1 = [1, 2, 3, 15000, 11000] # a, b, c 계수, startup, shutdown 비용
costCoeff_2 = [1, 2, 3, 15000, 11000] # a, b, c 계수, startup, shutdown 비용
costCoeff_3 = [1, 2, 3, 15000, 11000] # a, b, c 계수, startup, shutdown 비용

### 발전기 목록 생성 및 추가
generatorList = []
essList = []
#***중요 : 발전기 종류별로 최소 하나는 추가해야 함. 실제로 발전기가 없는 경우는 최소최대 발전을 0으로 설정 
generatorList.append( ThermalGenerator('신고리', 2, [2, 10], [1/60, 1/60], costCoeff, N_PIECE) )
essList.append( EnergyStorage('한전1', 4, [0, 1], [SOC_ESS_INIT, SOC_ESS_TERM], 5, 0.9) )

### Set number of variables
nGen = len(generatorList)
nEss = len(essList)
nTimeslot = len(load)


##### MILP #####
# 최적화 변수들에는 기존에 정한 naming rule을 따르지 않고 보편적으로 사용되는 변수 알파벳을 사용
# 변수 알파벳 옆의 아래 첨자는 '_'를 이용해 표시했고 아래 첨자는 가능한 naming rule을 따름

m = Model(name='Unit Commitment')

### 변수 생성
P_gen = m.continuous_var_list(nGen)
for i in range(nGen):
    P_gen[i] = m.continuous_var_cube(N_PIECE, nTimeslot, lb=0, ub=generatorList[i].maxPowerPerPiece)
P_gen = m.continuous_var_cube(nGen, N_PIECE, nTimeslot, lb=0, ub=[generatorList[i].maxPowerPerPiece for i in range()])
U_gen = m.binary_var_matrix(nGen, nTimeslot)
SU_gen = m.binary_var_matrix(nGen, nTimeslot)
SD_gen = m.binary_var_matrix(nGen, nTimeslot)

for i in range(nEss):
    P_essChg = m.continuous_var_matrix(nEss, nTimeslot, lb=0, ub=essList[i].maxPower)
    P_essDis = m.continuous_var_matrix(nEss, nTimeslot, lb=0, ub=essList[i].maxPower)
U_essChg = m.binary_var_matrix(nEss, nTimeslot)
U_essDis = m.binary_var_matrix(nEss, nTimeslot)

### 제약조건
# 모든전원 최소 최대 출력 제약, ONOFF 제약, 수급균형 제약
for j in range(nTimeslot):
    for i in range(nGen):
        m.add_constraint( sum(P_gen[i][k, j] for k in range(N_PIECE)) >= U_gen[i, j]*generatorList[i].minPower )
        m.add_constraint( sum(P_gen[i][k, j] for k in range(N_PIECE)) <= U_gen[i, j]*generatorList[i].maxPower )
        m.add_constraint( SU_gen[i, j] + SD_gen[i, j] <= 1 )
        m.add_constraint( sum(P_gen[i][k, j] for k in range(N_PIECE))
                          + sum(P_essDis[l, j] for l in range(nEss))
                          - sum(P_essChg[l, j] for l in range(nEss)) == load[j] )

    # for i in range(nEss):
    #     m.add_constraint( P_essDis[i, j] <= U_essDis[i, j]*essList[i].maxPower )
    #     m.add_constraint( P_essChg[i, j] <= U_essChg[i, j]*essList[i].maxPower )
    #     m.add_constraint( U_essDis[i, j] + U_essChg[i, j] <= 1 )

# SOC 제약

# for j in range(nTimeslot):
#     for i in range(nEss):
#         m.add_constraint( essList[i].initSOC
#                           - sum(P_essDis[i, k]*UNIT_TIME/essList[i].efficiency/essList[i].maxCapacity
#                                 for k in range(j + 1))
#                           + sum(P_essChg[i, k]*UNIT_TIME*essList[i].efficiency/essList[i].maxCapacity
#                                 for k in range(j + 1)) <=  SOC_ESS_MAX )
#         m.add_constraint( essList[i].initSOC
#                           - sum(P_essDis[i, k]*UNIT_TIME/essList[i].efficiency/essList[i].maxCapacity
#                                 for k in range(j + 1))
#                           + sum(P_essChg[i, k]*UNIT_TIME*essList[i].efficiency/essList[i].maxCapacity
#                                 for k in range(j + 1)) >=  SOC_ESS_MIN )
# for i in range(nEss):
#     m.add_constraint( essList[i].initSOC
#                      - sum(P_essDis[i, k]*UNIT_TIME/essList[i].efficiency/essList[i].maxCapacity for k in range(nTimeslot))
#                      + sum(P_essChg[i, k]*UNIT_TIME*essList[i].efficiency/essList[i].maxCapacity for k in range(nTimeslot))
#                      ==  essList[i].termSOC )

# 전원 rampup down 제약, ONOFF 제약
for j in range(nTimeslot - 1):
    # 발전기 rampup down 제약, ONOFF 제약
    for i in range(nGen):
        # m.add_constraint( sum(P_gen[i][k, j + 1] for k in range(N_PIECE))
        #                  - sum(P_gen[i][k, j] for k in range(N_PIECE)) <= generatorList[i].rampUpLimit*60*UNIT_TIME )
        # m.add_constraint( sum(P_gen[i][k, j + 1] for k in range(N_PIECE))
        #                  - sum(P_gen[i][k, j] for k in range(N_PIECE)) >= -generatorList[i].rampUpLimit*60*UNIT_TIME )
        m.add_constraint( U_gen[i, j + 1] - U_gen[i, j] == SU_gen[i, j + 1] - SD_gen[i, j + 1]  )
        
for i in range(nGen):
    m.add_constraint( U_gen[i, 0] == SU_gen[i, 0] ) # 시작할때부터 발전기 켜져있으면 startup 비용 추가

### 목적함수
m.set_objective("min", sum(generatorList[i].slopes[j]*P_gen[i][j, k] for i in range(nGen) for j in range(N_PIECE)
                           for k in range(nTimeslot))*UNIT_TIME
                + sum(U_gen[i, j]*generatorList[i].a*UNIT_TIME + SU_gen[i, j]*generatorList[i].startupCost
                      + SD_gen[i, j]*generatorList[i].shutdownCost for i in range(nGen) for j in range(nTimeslot)) )
                               
m.print_information()
sol = m.solve()
print(m.objective_value)

### Solution 저장
P_genSol = np.zeros([nGen, nTimeslot])
U_genSol = np.zeros([nGen, nTimeslot])
SU_genSol = np.zeros([nGen, nTimeslot])
SD_genSol = np.zeros([nGen, nTimeslot])
U_essDisSol = np.zeros([nEss, nTimeslot])
U_essChgSol = np.zeros([nEss, nTimeslot])

socEss = np.zeros([nEss, nTimeslot])
for i in range(nTimeslot):
    for j in range(nGen):
        P_genSol[j, i] = sum( sol.get_value(P_gen[j][k, i]) for k in range(N_PIECE) )
        U_genSol[j, i] = sol.get_value(U_gen[j, i])
        SU_genSol[j, i] = sol.get_value(SU_gen[j, i])
        SD_genSol[j, i] = sol.get_value(SD_gen[j, i])

    for j in range(nEss):
        P_essDisSol[j, i] = sol.get_value(P_essDis[j, i])
        P_essChgSol[j, i] = sol.get_value(P_essChg[j, i])
        U_essDisSol[j, i] = sol.get_value(U_essDis[j, i])
        U_essChgSol[j, i] = sol.get_value(U_essChg[j, i])
        socEss[j, i] = (essList[j].initSOC 
                        - sum(P_essDisSol[j, k]*UNIT_TIME/essList[j].efficiency/essList[j].maxCapacity for k in range(i + 1)) 
                        + sum(P_essChgSol[j, k]*UNIT_TIME*essList[j].efficiency/essList[j].maxCapacity for k in range(i + 1)))
