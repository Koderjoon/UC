__author__ = "Yun-Su Kim, Jun-Hyeok Kim, Jinsol Hwang, Author2"
__copyright__ = "Copyright 2022, Gwangju Institute of Science and Technology"
__credits__ = ["Yun-Su Kim", "Jun-Hyeok Kim", "Jinsol Hwang", "Author2"]
__version__ = "2.02"
__maintainer__ = "Jun-Hyeok Kim, Jinsol Hwang"
__email__ = "yunsukim@gist.ac.kr, junhyeok8407@gm.gist.ac.kr, j.s.hwang@gm.gist.ac.kr"
__status__ = "Test"


import numpy as np
import pandas as pd
import time

import UC_fig
import UC_model
from UC_MILP import MILP

start = time.time()

# SET SCENARIO TYPE
genCodeList = [4401, 2900, 2637, 7032, 7012, 2633, 2554]
# genCodeList = 'ALL'
if genCodeList == 'ALL':
    SCALE = 1
else:
    SCALE = 30

pumpCodeList = 'ALL'
# pumpCodeList = [1610, 1630, 1620, 1650, 1640, 1680, 1613]

cascading_Flag = False
fig_save_flag = True
csv_save_flag = True
tolerance = 0.0001
timelimit = 60*1
# GET LOAD PATTERN DATE WITH MAXIMUM GAP
loadPattern = pd.read_excel("./data/수요예측내역(22_4).xlsx", 0, header=3)
dload = (loadPattern['최대'] - loadPattern['최소'])[:-2]
repLoad = loadPattern.loc[dload.idxmax()]
load = repLoad[1:25]# - 25725.71 # 1시 - 24시
load = load / SCALE

pumpSOC = {'SOC_PUMP_MIN': 0.5, 'SOC_PUMP_MAX': 1.0, 'SOC_PUMP_INIT': 0.9, 'SOC_PUMP_TERM': 0.9}
essSOC = {'SOC_ESS_MIN': 0.3, 'SOC_ESS_MAX': 0.8, 'SOC_ESS_INIT': 0.5, 'SOC_ESS_TERM': 0.5}

ess1 = {'name': 'ESS1', 'busNumber': 3, 'minPower': 0, 'maxPower': 200, 'maxCapacity': 1000,
        'efficiency': 0.8, 'minSOC': essSOC['SOC_ESS_MIN'], 'maxSOC': essSOC['SOC_ESS_MAX'],
        'initSOC': essSOC['SOC_ESS_INIT'], 'termSOC': essSOC['SOC_ESS_TERM']}

ess2 = {'name': 'ESS2', 'busNumber': 3, 'minPower': 0, 'maxPower': 400, 'maxCapacity': 1600,
        'efficiency': 0.8, 'minSOC': essSOC['SOC_ESS_MIN'], 'maxSOC': essSOC['SOC_ESS_MAX'],
        'initSOC': essSOC['SOC_ESS_INIT'], 'termSOC': essSOC['SOC_ESS_TERM']}
essGroup = [ess1, ess2]

UNIT_TIME = 1  # Timeslot마다 에너지MWh를 구하기 위해 출력MW에 곱해져야하는 시간 단위, 1/4 = 15 min
N_PIECE = 3  # cost function 선형화 구간 수

# 예비력 요구량: 비중앙급전발전기 고려X, 양수ALL 고려X시 예비력 충분치 않아 경우 수렴 안함 (특히 Nspin)
# revReqDict = {'AGC_FC': 700/SCALE  , 'GF': 1000/SCALE ,
#               'AGC_sec': 1400/SCALE ,'Spin': 1400/SCALE ,'Nspin': 2000/SCALE}
revReqDict = {'REQ_reg': 700/SCALE, 'REQ_pri': 1000/SCALE,
              'REQ_sec': 1400/SCALE, 'REQ_ter': 1400/SCALE, 'REQ_seq': 2000/SCALE}

modelList = UC_model.GetModel(N_PIECE, genCodeList, pumpCodeList, pumpSOC, essGroup)
modelDict = modelList.ReadModel()

# for i in range(len(pumpList)):
#     pumpList[i].minPower = 0
# modelDict['pump'][0].fixedPumpPower = 40
# modelDict['pump'][1].fixedPumpPower = 60

modelList.printModelparameters(modelDict)

bus = np.loadtxt('./data/bus.txt', delimiter=',', skiprows=1, dtype=float)
branch = np.loadtxt('./data/branch.txt', delimiter=',', skiprows=1, dtype=float)

violationList = []
stop_flag = False
while not stop_flag:
    UC = MILP('Unit Commitment', modelDict, load, UNIT_TIME, N_PIECE, bus, branch,
              flow_limit=300, nuclear_flag=False)
    UC.constraint_minMax_balance()
    UC.constraint_SOC()
    UC.constraint_rampUpDown()
    UC.constraint_reserve()
    UC.constraint_reserve_req(revReqDict, cascading_flag=cascading_Flag)
    UC.constraint_gen_initState()
    # UC.cal_violation(violationList)
    UC.set_objective()
    sol = UC.solve(tolerance, timelimit)
    P_sol, U_sol, SoC_sol = UC.get_sol(sol)
    revSolDict = UC.get_sol_reserve(sol)

    end = time.time()
    print(f"Simulation Time : {round((end - start) / 60, 2)} min ")

    totalGen_Flag = genCodeList == 'ALL'
    totalPump_Flag = pumpCodeList == 'ALL'
    if csv_save_flag:
        UC.save_sol(repLoad[0], P_sol, revSolDict, totalGen_Flag)

    fig = UC_fig.UC_plot(UNIT_TIME, modelDict, load)
    fig.make_plot(P_sol, SoC_sol, totalGen_Flag, totalPump_Flag, save_flag=fig_save_flag)
    fig.make_res_plot(revSolDict, revReqDict, totalGen_Flag, totalPump_Flag, save_flag=fig_save_flag)
    fig.make_res_cascading(revSolDict, revReqDict, cascading_Flag, totalGen_Flag, totalPump_Flag, save_flag=fig_save_flag)

    stop_flag = True

    # violationList, cnt_violation = UC.check_flow_limit(violationList, P_sol)
    # print(violationList)
    # if cnt_violation == 0:
    #     stop_flag = True

# end = time.time()
# print(f"Simulation Time : {round((end - start)/60, 2)} min ")

# fig = UC_fig.UC_plot(UNIT_TIME, modelDict, load)
# fig.make_plot(P_sol, SoC_sol, totalGen_Flag, totalPump_Flag, save_flag=fig_save_flag)
# fig.make_res_plot(revSolDict, revReqDict, totalGen_Flag, save_flag=fig_save_flag)
# fig.make_res_cascading(revSolDict, revReqDict, totalGen_Flag, cascading_flag=cascading_Flag, save_flag=fig_save_flag)
