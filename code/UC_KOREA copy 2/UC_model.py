import pandas as pd
import numpy as np

### 발전기 객체 모델 / 단위: Power in MW, Ramp in MW/min
class ThermalGenerator:
    def __init__(self, gen, numberOfPieces):
        if type(gen['maxGF']) != int:
            gen['maxGF'] = gen['maxPower']
        if type(gen['maxLFC']) != int:
            gen['maxLFC'] = gen['maxPower']
        if type(gen['minGF']) != int:
            gen['minGF'] = gen['minPower']
        if type(gen['minLFC']) != int:
            gen['minLFC'] = gen['minPower']
        self.name = gen['name']
        self.busNumber = gen['busNumber']
        self.minPower = gen['minPower']
        self.maxPower = gen['maxPower']
        self.rampUpLimit = gen['rampUpLimit']
        self.rampDownLimit = gen['rampDownLimit']
        self.a = gen['상수']
        self.startupCost = gen['startUpCost']
        self.shutdownCost = gen['shutDownCost']
        self.fuelCost = gen['fuelCost']
        self.minLFC = gen['minLFC']
        self.maxLFC = gen['maxLFC']
        self.AVAC = gen['maxPower']  # 최대발전용량으로 일단
        self.minAGC = gen['minLFC']
        self.maxAGC = gen['maxLFC']
        self.minGF = gen['minGF']
        self.maxGF = gen['maxGF']
        self.GFRQ = gen['GFRQ']

        slopesOfCostFunction = np.zeros(numberOfPieces)
        self.maxPowerPerPiece = self.maxPower / numberOfPieces
        for i in range(numberOfPieces):
            slopesOfCostFunction[i] = (gen['상수'] + gen['1차 계수'] * (self.maxPowerPerPiece * (i + 1))
                                       + gen['2차 계수'] * (self.maxPowerPerPiece * (i + 1)) ** 2
                                       - gen['상수'] - gen['1차 계수'] * self.maxPowerPerPiece * i
                                       - gen['2차 계수'] * (self.maxPowerPerPiece * i) ** 2) / self.maxPowerPerPiece
        self.slopes = slopesOfCostFunction


class NuclearPlant:
    def __init__(self, nuclear, numberOfPieces):
        if type(nuclear['maxGF']) != int:
            nuclear['maxGF'] = nuclear['maxPower']
        if type(nuclear['maxLFC']) != int:
            nuclear['maxLFC'] = nuclear['maxPower']
        if type(nuclear['minGF']) != int:
            nuclear['minGF'] = nuclear['minPower']
        if type(nuclear['minLFC']) != int:
            nuclear['minLFC'] = nuclear['minPower']
        self.name = nuclear['name']
        self.busNumber = nuclear['busNumber']
        self.minPower = nuclear['minPower']
        self.maxPower = nuclear['maxPower']
        self.rampUpLimit = nuclear['rampUpLimit']
        self.rampDownLimit = nuclear['rampDownLimit']
        self.a = nuclear['상수']
        self.startupCost = nuclear['startUpCost']
        self.shutdownCost = nuclear['shutDownCost']
        self.fuelCost = nuclear['fuelCost']
        self.AVAC = nuclear['maxPower']  # 최대발전용량으로 일단

        slopesOfCostFunction = np.zeros(numberOfPieces)
        self.maxPowerPerPiece = self.maxPower / numberOfPieces
        for i in range(numberOfPieces):
            slopesOfCostFunction[i] = (nuclear['상수'] + nuclear['1차 계수'] * (self.maxPowerPerPiece * (i + 1))
                                       + nuclear['2차 계수'] * (self.maxPowerPerPiece * (i + 1)) ** 2
                                       - nuclear['상수'] - nuclear['1차 계수'] * self.maxPowerPerPiece * i
                                       - nuclear['2차 계수'] * (self.maxPowerPerPiece * i) ** 2) / self.maxPowerPerPiece
        self.slopes = slopesOfCostFunction


class PumpedStorage:
    def __init__(self, pump):
        self.name = pump['name']
        self.busNumber = pump['busNumber']
        self.minPower = pump['minPower']
        self.maxPower = pump['maxPower']
        self.minPump = 0  # Pump Data 획득 필요
        self.maxPump = pump['maxPump']  # Pump Data 획득 필요
        self.rampUpLimit = pump['rampUpLimit']
        self.rampDownLimit = pump['rampDownLimit']
        self.a = pump['상수']
        self.startupCost = pump['startUpCost']
        self.shutdownCost = pump['shutDownCost']
        self.fuelCost = pump['fuelCost']
        self.minLFC = pump['minLFC']
        self.maxLFC = pump['maxLFC']
        self.AVAC = pump['maxPower']  # 최대발전용량으로 일단
        self.minAGC = pump['minLFC']  # 제대로 계산 필요
        self.maxAGC = pump['maxLFC']  # 제대로 계산 필요
        self.minGF = pump['minGF']
        self.maxGF = pump['maxGF']
        self.GFRQ = pump['GFRQ']
        self.initSOC = pump['initSOC']
        self.termSOC = pump['termSOC']
        self.minSOC = pump['minSOC']
        self.maxSOC = pump['maxSOC']
        self.maxCapacity = pump['maxCapacity']
        self.efficiency = pump['efficiency']
        self.isFixedSpeed = pump['isFixedSpeed']
        self.fixedPumpPower = pump['fixedPumpPower']


class EnergyStorage:
    def __init__(self, ess):
        self.name = ess['name']
        self.busNumber = ess['busNumber']
        self.minPower = ess['minPower']
        self.maxPower = ess['maxPower']
        self.initSOC = ess['initSOC']
        self.termSOC = ess['termSOC']
        self.minSOC = ess['minSOC']
        self.maxSOC = ess['maxSOC']
        self.maxCapacity = ess['maxCapacity']
        self.efficiency = ess['efficiency']


class GetModel:
    def __init__(self, N_PIECE, genCodeList, pumpCodeList, pumpSOC, essGroup):
        self.N_PIECE = N_PIECE
        self.genCodeList = genCodeList
        self.pumpCodeList = pumpCodeList
        self.pumpSOC = pumpSOC
        self.essGroup = essGroup

    def ReadModel(self):
        genList = []
        nuclearList = []
        genEmptyList = []

        rscGenTable = pd.read_excel("./data/발전기자료_2018년8월RSC자료(최종).xlsx", sheet_name='최종', header=3)
        if self.genCodeList == 'ALL':
            genList, genEmptyList0 = self.ReadGeneratorFile(genList, rscGenTable, self.genCodeList, '기력,내연')
            genList, genEmptyList1 = self.ReadGeneratorFile(genList, rscGenTable, self.genCodeList, '복합_CC')
            nuclearList, genEmptyList2 = self.ReadGeneratorFile(nuclearList, rscGenTable, self.genCodeList, '원자력')
            genList, genEmptyList3 = self.ReadGeneratorFile(genList, rscGenTable, self.genCodeList, '수력')
            genEmptyList = genEmptyList0 + genEmptyList1 + genEmptyList2 + genEmptyList3
        else:
            genList, genCodeList = self.ReadGeneratorFile(genList, rscGenTable, self.genCodeList, '기력,내연')
            genList, genCodeList = self.ReadGeneratorFile(genList, rscGenTable, genCodeList, '복합_CC')
            #genList, genCodeList = self.ReadGeneratorFile(genList, rscGenTable, genCodeList, '복합_GT')
            nuclearList, genCodeList = self.ReadGeneratorFile(nuclearList, rscGenTable, genCodeList, '원자력')
            genList, genEmptyList = self.ReadGeneratorFile(genList, rscGenTable, genCodeList, '수력')

        if genEmptyList:
            genEmptyList = list(set(genEmptyList))
            print(f"Gen code does not matching: {genEmptyList}")

        pumpEmptyList = []
        pumpList, pumpEmptyList = self.ReadPumpStorageFile(rscGenTable, self.pumpCodeList, self.pumpSOC)

        if pumpEmptyList:
            pumpEmptyList = list(set(pumpEmptyList))
            print(f"Pump code does not matching: {pumpEmptyList}")

        essList = []
        for ess in self.essGroup:
            essList.append(EnergyStorage(ess))

        # if self.genCodeList != 'All':
            # print("ThermalGenerator: "+' '.join(genList[i].name for i in range(len(genList))))
            # print("NuclearPlant: "+' '.join(nuclearList[i].name for i in range(len(nuclearList))))
            # print("PumpedStorage: "+' '.join(pumpList[i].name for i in range(len(pumpList))))
            # print("EnergyStorage:"+' '.join(essList[i].name for i in range(len(essList))))
        
        modelDict = {'gen': genList, 'nuclear': nuclearList, 'pump': pumpList, 'ess': essList}
        return modelDict

    def ReadGeneratorFile(self, genList, rscGenTable, genCodeList, GenType):
        if GenType == '기력,내연':
            typeNum = 0
        elif GenType == '복합_CC':
            typeNum = 1
            df_GT = pd.read_excel("data/발전기자료_기술적특성자료.xlsx", 2, header=4)
        elif GenType == '복합_GT':
            typeNum = 2
        elif GenType == '원자력':
            typeNum = 3
        elif GenType == '수력':
            typeNum = 4

        techGenTable = pd.read_excel("data/발전기자료_기술적특성자료.xlsx", typeNum, header=4)

        if typeNum == 2:
            genCodeTable = techGenTable['대표GT'].to_numpy(dtype=int)
        elif typeNum == 4:
            techGenTable = techGenTable[techGenTable['발전원'] == '수력']
            genCodeTable = techGenTable['발전소'].to_numpy(dtype=int)
        else:
            genCodeTable = techGenTable['코드'].to_numpy(dtype=int)

        genCodeTable = genCodeTable[genCodeTable >= 0]
        gen = {}
        genEmptyList = []

        # Make Generator List
        if genCodeList == 'ALL':
            genCodeList = genCodeTable

        for genCode in genCodeList:
            if (rscGenTable['코드'] == genCode).any():
                rscGen = rscGenTable[rscGenTable['코드'] == genCode].iloc[0]
            else:
                genEmptyList.append(genCode)

            if genCode in genCodeTable:

                if typeNum != 4:
                    techGen = techGenTable[techGenTable['코드'] == genCode].iloc[0]
                else:
                    techGen = techGenTable[techGenTable['발전소'] == genCode].iloc[0]

                gen['name'] = techGen['발전기명']
                gen['busNumber'] = 2  # 임의값

                if typeNum in [1, 2]:
                    gen['minPower'] = techGen['최소발전용량(MW)']
                    gen['maxPower'] = techGen['최대발전용량(MW)']
                else:
                    gen['minPower'] = techGen['최소발전용량']
                    gen['maxPower'] = techGen['최대발전용량']

                gen['rampUpLimit'] = techGen['출력증가율']
                gen['rampDownLimit'] = techGen['출력감소율']

                gen['상수'] = rscGen['상수']
                gen['1차 계수'] = rscGen['1차 계수']
                gen['2차 계수'] = rscGen['2차 계수']
                gen['startUpCost'] = rscGen['HOT 기동\n비용(천원)']
                gen['shutDownCost'] = 0
                gen['fuelCost'] = rscGen['열량단가\n(천원/Gcal)']

                gen['최소운전시간'] = rscGen['최소운전\n시간(Hr)']
                gen['최소정지시간'] = rscGen['최소정지\n시간(Hr)']

                if typeNum in [1, 2]:
                    gen['maxGF'] = techGen['GF 상한(MW)']
                    gen['minGF'] = techGen['GF 하한(MW)']
                    gen['maxLFC'] = techGen['AGC 상한(MW)']
                    gen['minLFC'] = techGen['AGC 하한(MW)']

                else:
                    gen['maxGF'] = techGen['GF 상한']
                    gen['minGF'] = techGen['GF 하한']
                    gen['maxLFC'] = techGen['AGC 상한']
                    gen['minLFC'] = techGen['AGC 하한']

                if typeNum == 1:
                    if np.isnan(techGen['GFRQ']):
                        gen['GFRQ'] = round(sum(df_GT[df_GT['CC코드'] == genCode]['GFRQ']), 2)
                    else:
                        gen['GFRQ'] = techGen['GFRQ']
                elif typeNum != 3:
                    gen['GFRQ'] = techGen['GFRQ']

                if typeNum == 3:
                    genList.append(NuclearPlant(gen, self.N_PIECE))  # busnumber,maxSpinNspin 임의값
                else:
                    genList.append(ThermalGenerator(gen, self.N_PIECE))  # busnumber,maxSpinNspin 임의값
            else:
                genEmptyList.append(genCode)

        return genList, genEmptyList

    def ReadPumpStorageFile(self, rscGenTable, pumpCodeList, pumpSOC):
        techGenTable = pd.read_excel("data/발전기자료_기술적특성자료.xlsx", 4, header=4)
        techGenTable = techGenTable[techGenTable['발전원'] == '양수']

        pumpCodeTable = techGenTable['발전소'].to_numpy(dtype=int)
        pumpCodeTable = pumpCodeTable[pumpCodeTable >= 0]

        pump = {}
        pumpList = []
        pumpEmptyList = []

        # Make Generator List
        if pumpCodeList == 'ALL':
            pumpCodeTable = techGenTable['호기'].to_numpy(dtype=int)
            pumpCodeTable = pumpCodeTable[pumpCodeTable >= 0]
            pumpCodeList = pumpCodeTable
            total_flag = True
        else:
            total_flag = False

        # 발전기 단위
        for pumpCode in pumpCodeList:
            if total_flag:
                if (rscGenTable['코드'] == techGenTable[techGenTable['호기'] == pumpCode]['발전소'].values[0]).any():
                    rscGen = rscGenTable[rscGenTable['코드'] == techGenTable[techGenTable['호기'] == pumpCode]['발전소'].values[0]].iloc[0]
                else:
                    pumpEmptyList.append(pumpCode)
            else:
                if (rscGenTable['코드'] == pumpCode).any():
                    rscGen = rscGenTable[rscGenTable['코드'] == pumpCode].iloc[0]
                else:
                    pumpEmptyList.append(pumpCode)

            if pumpCode in pumpCodeTable:
                if total_flag:
                    techGen = techGenTable[techGenTable['호기'] == pumpCode]
                    pump['name'] = techGen['발전기명'].iloc[0]
                else:
                    techGen = techGenTable[techGenTable['발전소'] == pumpCode]
                    pump['name'] = techGen['발전소명'].iloc[0]

                pump['busNumber'] = 2  # 임의값

                pump['minPower'] = techGen['최소발전용량'].iloc[0]
                pump['maxPower'] = sum(techGen['최대발전용량'])

                pump['rampUpLimit'] = techGen['출력증가율'].iloc[0]
                pump['rampDownLimit'] = techGen['출력감소율'].iloc[0]

                pump['상수'] = rscGen['상수']
                pump['1차계수'] = rscGen['1차 계수']
                pump['2차 계수'] = rscGen['2차 계수']
                pump['startUpCost'] = rscGen['HOT 기동\n비용(천원)']
                pump['shutDownCost'] = 0
                pump['fuelCost'] = rscGen['열량단가\n(천원/Gcal)']

                pump['최소운전시간'] = rscGen['최소운전\n시간(Hr)']
                pump['최소정지시간'] = rscGen['최소정지\n시간(Hr)']

                pump['maxGF'] = sum(techGen['GF 상한'])
                pump['minGF'] = techGen['GF 하한'].iloc[0]
                pump['GFRQ'] = sum(techGen['GFRQ'])
                pump['maxLFC'] = sum(techGen['AGC 상한'])
                pump['minLFC'] = techGen['AGC 하한'].iloc[0]

                pump['initSOC'] = pumpSOC['SOC_PUMP_INIT']
                pump['termSOC'] = pumpSOC['SOC_PUMP_TERM']
                pump['minSOC'] = pumpSOC['SOC_PUMP_MIN']
                pump['maxSOC'] = pumpSOC['SOC_PUMP_MAX']

                pump['isFixedSpeed'] = 1

                if techGen['발전소명'].iloc[0] == '예천양수':
                    pump['maxPump'] = 430 * len(techGen)
                    pump['efficiency'] = 0.842
                    pump['maxCapacity'] = round(sum(techGen['설비용량']) * 10.2, 2)
                elif techGen['발전소명'].iloc[0] == '청송양수':
                    pump['maxPump'] = 320 * len(techGen)
                    pump['efficiency'] = 0.816
                    pump['maxCapacity'] = round(sum(techGen['설비용량']) * 10.1, 2)
                elif techGen['발전소명'].iloc[0] == '양양양수':
                    pump['maxPump'] = 265 * len(techGen)
                    pump['efficiency'] = 0.800
                    pump['maxCapacity'] = round(sum(techGen['설비용량']) * 11.2, 2)
                elif techGen['발전소명'].iloc[0] == '산청양수':
                    pump['maxPump'] = 375 * len(techGen)
                    pump['efficiency'] = 0.787
                    pump['maxCapacity'] = round(sum(techGen['설비용량']) * 10.4, 2)
                elif techGen['발전소명'].iloc[0] == '무주양수':
                    pump['maxPump'] = 330 * len(techGen)
                    pump['efficiency'] = 0.765
                    pump['maxCapacity'] = round(sum(techGen['설비용량']) * 8.4, 2)
                elif techGen['발전소명'].iloc[0] == '삼량진양수':
                    pump['maxPump'] = 360 * len(techGen)
                    pump['efficiency'] = 0.763
                    pump['maxCapacity'] = round(sum(techGen['설비용량']) * 7.3, 2)
                elif techGen['발전소명'].iloc[0] == '청평양수':
                    pump['maxPump'] = 200 * len(techGen)
                    pump['efficiency'] = 0.723
                    pump['maxCapacity'] = round(sum(techGen['설비용량']) * 8.0, 2)

                pump['fixedPumpPower'] = round(pump['maxPump'] / 6, 2)

                pumpList.append(PumpedStorage(pump))  # busnumber,maxSpinNspin 임의값
            else:
                pumpEmptyList.append(pumpCode)

        return pumpList, pumpEmptyList

    def printModelparameters(self, modelDict):
        
        genList = modelDict['gen']
        nuclearList = modelDict['nuclear']
        pumpList = modelDict['pump']
        essList = modelDict['ess']
        
        if self.genCodeList == 'ALL':
            print(
                "ALL GenList"
                )    
        else: 
            for i in range(len(genList)):
                print(
                    f"ThermalGenerator({genList[i].name}, busNumber={genList[i].busNumber}, minMaxPower={[genList[i].minPower,genList[i].maxPower]},"
                    f" rampUpDownLimit={[genList[i].rampUpLimit,genList[i].rampDownLimit]},"
                    f" startUpDownCost={[genList[i].startupCost,genList[i].shutdownCost]}, fuelCost={genList[i].fuelCost},"
                    f" minMaxLFC={[genList[i].minLFC, genList[i].maxLFC]}, minMaxAGC={[genList[i].minAGC, genList[i].maxAGC]},"
                    f" minMaxGF={[genList[i].minGF, genList[i].maxGF]}, AVAC={genList[i].AVAC}, GFRQ={genList[i].GFRQ})"
                )
            for i in range(len(nuclearList)):
                print(
                    f"NuclearPlant({nuclearList[i].name}, busNumber={nuclearList[i].busNumber}, minMaxPower={[nuclearList[i].minPower, nuclearList[i].maxPower]},"
                    f" rampUpDownLimit={[nuclearList[i].rampUpLimit, nuclearList[i].rampDownLimit]},"
                    f" startUpDownCost={[nuclearList[i].startupCost,nuclearList[i].shutdownCost]}, fuelCost={nuclearList[i].fuelCost},"
                    f" AVAC={nuclearList[i].AVAC})"
                )
        for i in range(len(pumpList)):
            print(
                f"PumpStorage({pumpList[i].name}, busNumber={pumpList[i].busNumber}, minMaxPowerPump={[pumpList[i].minPower,pumpList[i].maxPower,pumpList[i].minPump,pumpList[i].maxPump]},"
                f" rampUpDownLimit={[pumpList[i].rampUpLimit,pumpList[i].rampDownLimit]}, initTermSOC={[pumpList[i].initSOC,pumpList[i].termSOC]}, minMaxSOC={[pumpList[i].minSOC, pumpList[i].maxSOC]},"
                f" maxCapacity={pumpList[i].maxCapacity}, efficiency={pumpList[i].efficiency}, isFixedSpeed={pumpList[i].isFixedSpeed}, fixedPumpPower={pumpList[i].fixedPumpPower},"
                f" minMaxLFC={[pumpList[i].minLFC, pumpList[i].maxLFC]}, minMaxAGC={[pumpList[i].minAGC, pumpList[i].maxAGC]},"
                f" minMaxGF={[pumpList[i].minGF, pumpList[i].maxGF]}, AVAC={pumpList[i].AVAC}, GFRQ={pumpList[i].GFRQ})"
            )
        for i in range(len(essList)):
            print(
                f"EnergyStorage({essList[i].name}, busNumber={essList[i].busNumber}, minMaxPower={[essList[i].minPower,essList[i].maxPower]},"
                f" initTermSOC={[essList[i].initSOC,essList[i].termSOC]}, minMaxSOC={[essList[i].minSOC,essList[i].maxSOC]}, "
                f"maxCapacity={essList[i].maxCapacity}, efficiency={essList[i].efficiency})"
            )
