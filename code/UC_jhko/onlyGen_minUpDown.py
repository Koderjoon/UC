import numpy as np
import matplotlib.pyplot as plt
from docplex.mp.model import Model
from gen_class_1 import *
### Load data
load = np.array([12, 12, 12, 12, 46.0367, 49.0137, 64.8265, 85.3439, 12, 12, 12, 114.693, 106.74, 103.513, 103.224, 109.631, 120.798, 130.971, 120.982, 110.441, 99.9834, 83.1814, 64.4019, 54.2642,46.4251, 12, 12, 41.4513, 46.0367, 49.0137, 64.8265, 85.3439, 95.3564, 104.801, 111.623, 114.693, 106.74, 103.513, 103.224, 109.631, 120.798, 130.971, 120.982, 110.441, 99.9834, 130.971, 120.982, 110.441, 99.9834, 83.1814, 64.4019, 54.2642, 46.4251, 42.8597, 46.0367, 49.0137, 64.8265, 85.3439, 95.3564, 104.801, 111.623, 114.693, 106.74, 103.513, 103.224, 85.3439, 95.3564, 104.801, 111.623, 12, 12, 103.513, 103.224])
load = load*3
UNIT_TIME = 1 # Timeslot마다 에너지MWh를 구하기 위해 출력MW에 곱해져야하는 시간 단위, 1/4 = 15 min
N_PIECE = 3 # cost function을 선형화할 구간 수
DIESEL_COST = 1200
MIN_ON_TIME = 10
MIN_OFF_TIME = 5

costCoeff_1 = [5.9, 0.15, 0.001, 15000, 11000] # a, b, c 계수, startup, shutdown 비용
costCoeff_2 = [5.85, 0.145, 0.0011, 15000, 11000] # a, b, c 계수, startup, shutdown 비용
costCoeff_3 = [5.95, 0.14, 0.001, 15000, 11000] # a, b, c 계수, startup, shutdown 비용

### 발전기 목록 생성 및 추가
generatorList = []

#***중요 : 발전기 종류별로 최소 하나는 추가해야 함. 실제로 발전기가 없는 경우는 최소최대 발전을 0으로 설정 
generatorList.append( ThermalGenerator('Gen1', [10, 140], costCoeff_1, N_PIECE) )
generatorList.append( ThermalGenerator('Gen2', [10, 140], costCoeff_2, N_PIECE) )
generatorList.append( ThermalGenerator('Gen3', [10, 140], costCoeff_3, N_PIECE) )

### Set number of variables
nGen = len(generatorList)

nTimeslot = len(load)


##### MILP #####
# 최적화 변수들에는 기존에 정한 naming rule을 따르지 않고 보편적으로 사용되는 변수 알파벳을 사용
# 변수 알파벳 옆의 아래 첨자는 '_'를 이용해 표시했고 아래 첨자는 가능한 naming rule을 따름

m = Model(name='Unit Commitment')

### 변수 생성
P_gen = m.continuous_var_cube(nGen, N_PIECE, nTimeslot, lb=0,
                                ub=[generatorList[i].maxPowerPerPiece for i in range(nGen) 
                                for _ in range(N_PIECE) for _ in range(nTimeslot)])
U_gen = m.binary_var_matrix(nGen, nTimeslot)
SU_gen = m.binary_var_matrix(nGen, nTimeslot)
SD_gen = m.binary_var_matrix(nGen, nTimeslot)



### 제약조건
# 모든전원 최소 최대 출력 제약, ONOFF 제약, 수급균형 제약
for j in range(nTimeslot):
    for i in range(nGen):
        m.add_constraint( sum(P_gen[i, k, j] for k in range(N_PIECE)) >= U_gen[i, j]*generatorList[i].minPower )
        m.add_constraint( sum(P_gen[i, k, j] for k in range(N_PIECE)) <= U_gen[i, j]*generatorList[i].maxPower )
        m.add_constraint( SU_gen[i, j] + SD_gen[i, j] <= 1 )
    m.add_constraint( sum(P_gen[i, k, j] for k in range(N_PIECE) for i in range(nGen)) == load[j])



# ONOFF 제약
for j in range(nTimeslot - 1):
    for i in range(nGen):
        m.add_constraint( U_gen[i, j + 1] - U_gen[i, j] == SU_gen[i, j + 1] - SD_gen[i, j + 1]  )
        
for i in range(nGen):
    m.add_constraint( U_gen[i, 0] == SU_gen[i, 0] ) # 시작할때부터 발전기 켜져있으면 startup 비용 추가

#최소운전/정지시간
for i in range(nGen):
    for k in range(nTimeslot):
        if k <= nTimeslot-MIN_ON_TIME:
            m.add_constraint(sum(U_gen[i,j] for j in range(k,k+MIN_ON_TIME)) >= MIN_ON_TIME*SU_gen[i,k])
        else: 
            m.add_constraint(sum(U_gen[i,j] for j in range(k,nTimeslot)) >= MIN_ON_TIME*SU_gen[i,k])

        if k <= nTimeslot-MIN_OFF_TIME:
            m.add_constraint(sum((1-U_gen[i,j]) for j in range(k,k+MIN_OFF_TIME)) >= MIN_OFF_TIME*SD_gen[i,k])
        else: 
            m.add_constraint(sum((1-U_gen[i,j]) for j in range(k,nTimeslot)) >= MIN_OFF_TIME*SD_gen[i,k])

### 목적함수
m.set_objective("min", sum(generatorList[i].slopes[j]*P_gen[i, j, k]*DIESEL_COST*UNIT_TIME for i in range(nGen) for j in range(N_PIECE)
                           for k in range(nTimeslot)))
                # + sum(U_gen[i, j]*generatorList[i].a*UNIT_TIME*DIESEL_COST + SU_gen[i, j]*generatorList[i].startupCost
                #        + SD_gen[i, j]*generatorList[i].shutdownCost for i in range(nGen) for j in range(nTimeslot)) )
                               
m.print_information()
sol = m.solve()
if sol is None:
    print('- model is infeasible')
print(m.objective_value)

### Solution 저장
P_genSol = np.zeros([nGen, nTimeslot])
U_genSol = np.zeros([nGen, nTimeslot])
SU_genSol = np.zeros([nGen, nTimeslot])
SD_genSol = np.zeros([nGen, nTimeslot])

for i in range(nTimeslot):
    for j in range(nGen):
        P_genSol[j, i] = sum( sol.get_value(P_gen[j, k, i]) for k in range(N_PIECE) )
        U_genSol[j, i] = sol.get_value(U_gen[j, i])
        SU_genSol[j, i] = sol.get_value(SU_gen[j, i])
        SD_genSol[j, i] = sol.get_value(SD_gen[j, i])



plt.style.use('_mpl-gallery')

hours = np.arange(nTimeslot)
generation = {
    'gen1': P_genSol[0],
    'gen2': P_genSol[1],
    'gen3': P_genSol[2],
}

fig, ax = plt.subplots(figsize=(5, 2.7), layout='constrained')
ax.stackplot(hours, generation.values(),
             labels=generation.keys(), alpha=0.8)
ax.legend(loc='upper left')
ax.set_title('UC')
ax.set_xlabel('Time')
ax.set_ylabel('Load(kW)')

plt.show()

