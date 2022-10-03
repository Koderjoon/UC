import numpy as np
import matplotlib.pyplot as plt
import time

class UC_plot:
    def __init__(self, UNIT_TIME, modelDict, load):
        self.nTimeslot = len(load)
        self.UNIT_TIME = UNIT_TIME
        self.generatorList = modelDict['gen']
        self.nuclearList = modelDict['nuclear']
        self.pumpList = modelDict['pump']
        self.essList = modelDict['ess']
        self.nGen = len(self.generatorList)
        self.nNuc = len(self.nuclearList)
        self.nPump = len(self.pumpList)
        self.nEss = len(self.essList)
        self.load = load

    def make_plot(self, P_sol, SoC_sol, totalGen_Flag=False, totalPump_Flag=False, last_flag=False, save_flag=False):
        [P_genSol, P_nuclearSol, P_pumpDisSol, P_pumpChgSol, P_essDisSol, P_essChgSol] = P_sol
        [socPump, socESS] = SoC_sol

        plt.rcParams["figure.figsize"] = (6, 9)
        plt.rcParams['font.family'] = 'Malgun Gothic'
        plt.rcParams['axes.unicode_minus'] = False
        legend_fontsize = 6

        plt.figure()
        plt.subplot(311)
        plt.plot(np.arange(self.nTimeslot), self.load, label='Load', color='navy')
        if totalGen_Flag:
            self.nGen = 1
            plt.plot(np.arange(self.nTimeslot), sum(P_genSol), label='Gen_sum')
            plt.plot(np.arange(self.nTimeslot), sum(P_nuclearSol), label='Nuclear_sum', color='purple')
        else:
            for i in range(self.nGen):
                plt.plot(np.arange(self.nTimeslot), P_genSol[i, :], label=self.generatorList[i].name)
            for i in range(self.nNuc):
                plt.plot(np.arange(self.nTimeslot), P_nuclearSol[i, :], label=self.nuclearList[i].name,
                         color='C' + str(i + self.nGen + self.nPump + self.nEss))
        plt.legend(loc='best', ncol=3, fontsize=legend_fontsize)
        plt.grid(alpha=0.4, linestyle='--')
        plt.xticks(np.arange(0, self.nTimeslot, 1 / self.UNIT_TIME), fontsize=8)
        plt.ylabel('Load & Gen (MW)')
        plt.xlim([0, self.nTimeslot - 1])
        plt.ylim([0, max(self.load) * 1.1])

        plt.subplot(312)
        if totalPump_Flag:
            plt.plot(np.arange(self.nTimeslot), sum(P_pumpDisSol) - sum(P_pumpChgSol), label='Pump_sum',
                     color='C' + str(self.nGen))
        else:
            for i in range(self.nPump):
                plt.plot(np.arange(self.nTimeslot), P_pumpDisSol[i, :] - P_pumpChgSol[i, :], label=self.pumpList[i].name,
                         color='C' + str(i + self.nGen))
        for i in range(self.nEss):
            plt.plot(np.arange(self.nTimeslot), P_essDisSol[i, :] - P_essChgSol[i, :], label=self.essList[i].name,
                     color='C' + str(i + self.nGen + self.nPump))
        
        plt.legend(loc='best', ncol=3, fontsize=legend_fontsize)
        plt.grid(alpha=0.4, linestyle='--')
        plt.xticks(np.arange(0, self.nTimeslot, 1 / self.UNIT_TIME), fontsize=8)
        plt.ylabel('PSH & ESS (MW)')
        plt.xlim([0, self.nTimeslot - 1])

        plt.subplot(313)
        if totalPump_Flag:
            plt.plot(np.arange(self.nTimeslot), sum(socPump)/self.nPump, label='Pump_sum',
                     color='C' + str(self.nGen))
        else:
            for i in range(self.nPump):
                plt.plot(np.arange(self.nTimeslot), socPump[i, :], label=self.pumpList[i].name,
                         color='C' + str(i + self.nGen))
        for i in range(self.nEss):
            plt.plot(np.arange(self.nTimeslot), socESS[i, :], label=self.essList[i].name,
                     color='C' + str(i + self.nGen + self.nPump))
        
        plt.legend(loc='best', ncol=3, fontsize=legend_fontsize)
        plt.grid(alpha=0.4, linestyle='--')
        plt.xticks(np.arange(0, self.nTimeslot, 1 / self.UNIT_TIME), fontsize=8)
        plt.ylim([0, 1])
        plt.ylabel('SoC')
        plt.xlim([0, self.nTimeslot - 1])
        plt.suptitle('UC results', y=0.91, fontsize=14)
        
        if save_flag:
            plt.savefig('./'+time.strftime("%m%d%H%M")+'_total.png', dpi=300, bbox_inches='tight')
        if last_flag:
            plt.show()

    def make_res_plot(self, reserveDict, revReqDict, total_flag=False, totalPump_Flag=False, last_flag=False, save_flag=False):
        [P_gen_pri, P_pump_pri] = reserveDict['RES_pri']
        [P_gen_reg, P_pump_reg] = reserveDict['RES_reg']
        [P_gen_sec, P_pump_sec] = reserveDict['RES_sec']
        [P_gen_ter, P_pump_terSpin, P_pump_terNspin] = reserveDict['RES_ter']
        [P_gen_seq, P_pump_seq] = reserveDict['RES_seq']

        req_pri = revReqDict['REQ_pri']
        req_reg = revReqDict['REQ_reg']
        req_sec = revReqDict['REQ_sec']
        req_ter = revReqDict['REQ_ter']
        req_seq = revReqDict['REQ_seq']

        plt.rcParams["figure.figsize"] = (6, 9)
        plt.rcParams['font.family'] = 'Malgun Gothic'
        plt.rcParams['axes.unicode_minus'] = False
        legend_fontsize = 6

        plt.figure()
        plt.subplot(511)
        if total_flag:
            sumP_gen_pri = sum(P_gen_pri)
            plt.bar(np.arange(self.nTimeslot), sumP_gen_pri, label="Gen1차_sum")
        else:
            for i in range(self.nGen):
                plt.bar(np.arange(self.nTimeslot), P_gen_pri[i, :], bottom=sum(P_gen_pri[i, :] for i in range(i)),
                        label=self.generatorList[i].name)
        if totalPump_Flag:
            sumP_pump_pri = sum(P_pump_pri)
            plt.bar(np.arange(self.nTimeslot), sumP_pump_pri, label="PSH1차_sum", bottom=sum(P_gen_pri))
        else:
            for j in range(self.nPump):
                plt.bar(np.arange(self.nTimeslot), P_pump_pri[j, :], bottom=sum(P_pump_pri[j, :] for j in range(j))+sum(P_gen_pri),
                        label=self.pumpList[j].name)
            
        plt.legend(loc='best', ncol=3, fontsize=legend_fontsize)
        plt.grid(alpha=0.4, linestyle='--')
        plt.xticks(np.arange(0, self.nTimeslot, 1 / self.UNIT_TIME), fontsize=8)
        plt.ylabel('1차예비력 (MW)')
        plt.xlim([-0.7, self.nTimeslot - 0.3])
        plt.axhline(req_pri, -0.7, self.nTimeslot - 0.3, color='k', linewidth=1)
        plt.ylim([0, max(sum(np.vstack((P_gen_pri, P_pump_pri))))*1.1])

        plt.subplot(512)
        if total_flag:
            sumP_gen_reg = sum(P_gen_reg)
            plt.bar(np.arange(self.nTimeslot), sumP_gen_reg, label="Gen주파수_sum")
        else:
            for i in range(self.nGen):
                plt.bar(np.arange(self.nTimeslot), P_gen_reg[i, :], bottom=sum(P_gen_reg[i, :] for i in range(i)),
                        label=self.generatorList[i].name)
        if totalPump_Flag:
            sumP_pump_reg = sum(P_pump_reg)
            plt.bar(np.arange(self.nTimeslot), sumP_pump_reg, label="PSH주파수_sum", bottom=sum(P_gen_reg))
        else:
            for j in range(self.nPump):
                plt.bar(np.arange(self.nTimeslot), P_pump_reg[j, :], bottom=sum(P_pump_reg[j, :] for j in range(j))+sum(P_gen_reg),
                        label=self.pumpList[j].name)
        plt.legend(loc='best', ncol=3, fontsize=legend_fontsize)
        plt.grid(alpha=0.4, linestyle='--')
        plt.xticks(np.arange(0, self.nTimeslot, 1 / self.UNIT_TIME), fontsize=8)
        plt.ylabel('주파수제어예비력 (MW)')
        plt.xlim([-0.7, self.nTimeslot - 0.3])
        plt.axhline(req_reg, -0.7, self.nTimeslot - 0.3, color='k', linewidth=1)
        plt.ylim([0, max(sum(np.vstack((P_gen_reg, P_pump_reg)))) * 1.1])

        plt.subplot(513)
        if total_flag:
            sumP_gen_sec = sum(P_gen_sec)
            plt.bar(np.arange(self.nTimeslot), sumP_gen_sec, label="Gen2차_sum")
        else:
            for i in range(self.nGen):
                plt.bar(np.arange(self.nTimeslot), P_gen_sec[i, :], bottom=sum(P_gen_sec[i, :] for i in range(i)),
                        label=self.generatorList[i].name)
        if totalPump_Flag:
            sumP_pump_sec = sum(P_pump_sec)
            plt.bar(np.arange(self.nTimeslot), sumP_pump_sec, label="PSH2차_sum", bottom=sum(P_gen_sec))
        else:
            for j in range(self.nPump):
                plt.bar(np.arange(self.nTimeslot), P_pump_sec[j, :], bottom=sum(P_pump_sec[j, :] for j in range(j))+sum(P_gen_sec),
                        label=self.pumpList[j].name)
        plt.legend(loc='best', ncol=3, fontsize=legend_fontsize)
        plt.grid(alpha=0.4, linestyle='--')
        plt.xticks(np.arange(0, self.nTimeslot, 1 / self.UNIT_TIME), fontsize=8)
        plt.ylabel('2차예비력 (MW)')
        plt.xlim([-0.7, self.nTimeslot - 0.3])
        plt.axhline(req_sec, -0.7, self.nTimeslot - 0.3, color='k', linewidth=1)
        plt.ylim([0, max(sum(np.vstack((P_gen_sec, P_pump_sec)))) * 1.1])

        plt.subplot(514)
        if total_flag:
            sumP_gen_ter = sum(P_gen_ter)
            plt.bar(np.arange(self.nTimeslot), sumP_gen_ter, label="Gen3차_sum")
        else:
            for i in range(self.nGen):
                plt.bar(np.arange(self.nTimeslot), P_gen_ter[i, :], bottom=sum(P_gen_ter[i, :] for i in range(i)),
                        label=self.generatorList[i].name)
        if totalPump_Flag:
            sumP_pump_terS = sum(P_pump_terSpin)
            sumP_pump_terN = sum(P_pump_terNspin)
            plt.bar(np.arange(self.nTimeslot), sumP_pump_terS, label="PSH3차(운전)_sum", bottom=sum(P_gen_ter))
            plt.bar(np.arange(self.nTimeslot), sumP_pump_terN, label="PSH3차(정지)_sum", bottom=sum(P_gen_ter)+sum(P_pump_terSpin))
        else:
            for j in range(self.nPump):
                plt.bar(np.arange(self.nTimeslot), P_pump_terSpin[j, :]+P_pump_terNspin[j, :], bottom=sum(P_pump_terSpin[j, :]+P_pump_terNspin[j, :] for j in range(j))+sum(P_gen_ter),
                        label=self.pumpList[j].name)
        plt.legend(loc='best', ncol=3, fontsize=legend_fontsize)
        plt.grid(alpha=0.4, linestyle='--')
        plt.xticks(np.arange(0, self.nTimeslot, 1 / self.UNIT_TIME), fontsize=8)
        plt.ylabel('3차예비력 (MW)')
        plt.xlim([-0.7, self.nTimeslot - 0.3])
        plt.axhline(req_ter, -0.7, self.nTimeslot - 0.3, color='k', linewidth=1)
        plt.ylim([0, max(sum(np.vstack((P_gen_ter, P_pump_terSpin, P_pump_terNspin)))) * 1.1])

        plt.subplot(515)
        if total_flag:
            sumP_gen_seq = sum(P_gen_seq)
            plt.bar(np.arange(self.nTimeslot), sumP_gen_seq, label="Gen속응성_sum")
        else:
            for i in range(self.nGen):
                plt.bar(np.arange(self.nTimeslot), P_gen_seq[i, :], bottom=sum(P_gen_seq[i, :] for i in range(i)),
                        label=self.generatorList[i].name)
        if totalPump_Flag:
            sumP_pump_seq = sum(P_pump_seq)
            plt.bar(np.arange(self.nTimeslot), sumP_pump_seq, label="PSH속응성_sum", bottom=sum(P_gen_seq))
        else:
            for j in range(self.nPump):
                plt.bar(np.arange(self.nTimeslot), P_pump_seq[j, :], bottom=sum(P_pump_seq[j, :] for j in range(j))+sum(P_gen_seq),
                        label=self.pumpList[j].name)
        plt.legend(loc='best', ncol=3, fontsize=legend_fontsize)
        plt.grid(alpha=0.4, linestyle='--')
        plt.xticks(np.arange(0, self.nTimeslot, 1 / self.UNIT_TIME), fontsize=8)
        plt.ylabel('속응성자원 (MW)')
        plt.xlim([-0.7, self.nTimeslot - 0.3])
        plt.axhline(req_seq, -0.7, self.nTimeslot - 0.3, color='k', linewidth=1)
        plt.ylim([0, max(sum(np.vstack((P_gen_seq, P_pump_seq)))) * 1.1])
        plt.suptitle('Reserve results', y=0.91, fontsize=14)
        
        if save_flag:
            plt.savefig('./'+time.strftime("%m%d%H%M")+'_reserve.png', dpi=300, bbox_inches='tight')
        if last_flag:
            plt.show()

    def make_res_cascading(self, reserveDict, revReqDict, cascading_flag, total_flag=False, totalPump_Flag=False, last_flag=True, save_flag=False):
        
        [P_gen_pri, P_pump_pri] = reserveDict['RES_pri']
        [P_gen_reg, P_pump_reg] = reserveDict['RES_reg']
        [P_gen_sec, P_pump_sec] = reserveDict['RES_sec']
        [P_gen_ter, P_pump_terSpin, P_pump_terNspin] = reserveDict['RES_ter']

        req_pri = revReqDict['REQ_pri']
        req_reg = revReqDict['REQ_reg']
        req_sec = revReqDict['REQ_sec']
        req_ter = revReqDict['REQ_ter']

        plt.rcParams['font.family'] = 'Malgun Gothic'
        plt.rcParams['axes.unicode_minus'] = False
        legend_fontsize = 6

        if cascading_flag:
            plt.rcParams["figure.figsize"] = (6, 9)
            plt.figure(3)
            plt.subplot(411)
            if total_flag:
                sumP_gen_pri = sum(P_gen_pri)
                plt.bar(np.arange(self.nTimeslot), sumP_gen_pri, label="Gen_sum")
            else:
                for i in range(self.nGen):
                    plt.bar(np.arange(self.nTimeslot), P_gen_pri[i, :], bottom=sum(P_gen_pri[i, :] for i in range(i)),
                            label=self.generatorList[i].name)
            if totalPump_Flag:
                sumP_pump_pri = sum(P_pump_pri)
                plt.bar(np.arange(self.nTimeslot), sumP_pump_pri, label="PSH_sum", bottom=sum(P_gen_pri))
            else:
                for j in range(self.nPump):
                    plt.bar(np.arange(self.nTimeslot), P_pump_pri[j, :],
                            bottom=sum(P_pump_pri[j, :] for j in range(j)) + sum(P_gen_pri),
                            label=self.pumpList[j].name)
            plt.legend(loc='best', ncol=3, fontsize=legend_fontsize)
            plt.grid(alpha=0.4, linestyle='--')
            plt.xticks(np.arange(0, self.nTimeslot, 1 / self.UNIT_TIME), fontsize=8)
            plt.ylabel('1차예비력 (MW)')
            plt.xlim([-0.7, self.nTimeslot - 0.3])
            plt.axhline(req_pri, -0.7, self.nTimeslot - 0.3, color='k', linewidth=1)
            plt.ylim([0, max(sum(np.vstack((P_gen_pri, P_pump_pri)))) * 1.1])

            plt.subplot(412)
            if total_flag:
                sumP_gen_pri = sum(P_gen_pri)
                sumP_gen_reg = sum(P_gen_reg)
                plt.bar(np.arange(self.nTimeslot), sumP_gen_pri + sumP_gen_reg, label="Gen_sum")
            else:
                for i in range(self.nGen):
                    plt.bar(np.arange(self.nTimeslot), P_gen_pri[i, :] + P_gen_reg[i, :],
                            bottom=sum(P_gen_pri[i, :] + P_gen_reg[i, :] for i in range(i)),
                            label=self.generatorList[i].name)
            if totalPump_Flag:
                sumP_pump_pri = sum(P_pump_pri)
                sumP_pump_reg = sum(P_pump_reg)
                plt.bar(np.arange(self.nTimeslot), sumP_pump_pri + sumP_pump_reg, label="PSH_sum",
                        bottom=sum(P_gen_pri + P_gen_reg))
            else:
                for j in range(self.nPump):
                    plt.bar(np.arange(self.nTimeslot), P_pump_pri[j, :] + P_pump_reg[j, :],
                            bottom=sum(P_pump_pri[j, :] + P_pump_reg[j, :] for j in range(j)) + sum(P_gen_pri + P_gen_reg),
                            label=self.pumpList[j].name)
            plt.legend(loc='best', ncol=3, fontsize=legend_fontsize)
            plt.grid(alpha=0.4, linestyle='--')
            plt.xticks(np.arange(0, self.nTimeslot, 1 / self.UNIT_TIME), fontsize=8)
            plt.ylabel('1차+주∙제 (MW)')
            plt.xlim([-0.7, self.nTimeslot - 0.3])
            plt.axhline(req_pri, -0.7, self.nTimeslot - 0.3, color='k', linewidth=1)
            plt.axhline(req_pri + req_reg, -0.7, self.nTimeslot - 0.3, color='k', linewidth=1)
            plt.ylim([0, max(sum(np.vstack((P_gen_pri + P_gen_reg, P_pump_pri + P_pump_reg)))) * 1.1])

            plt.subplot(413)
            if total_flag:
                sumP_gen_pri = sum(P_gen_pri)
                sumP_gen_reg = sum(P_gen_reg)
                sumP_gen_sec = sum(P_gen_sec)
                plt.bar(np.arange(self.nTimeslot), sumP_gen_pri + sumP_gen_reg + sumP_gen_sec, label="Gen_sum")
            else:
                for i in range(self.nGen):
                    plt.bar(np.arange(self.nTimeslot), P_gen_pri[i, :] + P_gen_reg[i, :] + P_gen_sec[i, :],
                            bottom=sum(P_gen_pri[i, :] + P_gen_reg[i, :] + P_gen_sec[i, :] for i in range(i)),
                            label=self.generatorList[i].name)
            if totalPump_Flag:
                sumP_pump_pri = sum(P_pump_pri)
                sumP_pump_reg = sum(P_pump_reg)
                sumP_pump_sec = sum(P_pump_sec)
                plt.bar(np.arange(self.nTimeslot), sumP_pump_pri + sumP_pump_reg + sumP_pump_sec, label="PSH_sum",
                        bottom=sum(P_gen_pri + P_gen_reg + P_gen_sec))
            else:
                for j in range(self.nPump):
                    plt.bar(np.arange(self.nTimeslot), P_pump_pri[j, :] + P_pump_reg[j, :] + P_pump_sec[j, :],
                            bottom=sum(P_pump_pri[j, :] + P_pump_reg[j, :] + P_pump_sec[j, :]
                                       for j in range(j)) + sum(P_gen_pri + P_gen_reg + P_gen_sec),
                            label=self.pumpList[j].name)
            plt.legend(loc='best', ncol=3, fontsize=legend_fontsize)
            plt.grid(alpha=0.4, linestyle='--')
            plt.xticks(np.arange(0, self.nTimeslot, 1 / self.UNIT_TIME), fontsize=8)
            plt.ylabel('1차+주∙제+2차 (MW)')
            plt.xlim([-0.7, self.nTimeslot - 0.3])
            plt.axhline(req_pri, -0.7, self.nTimeslot - 0.3, color='k', linewidth=1)
            plt.axhline(req_pri + req_reg, -0.7, self.nTimeslot - 0.3, color='k', linewidth=1)
            plt.axhline(req_pri + req_reg + req_sec, -0.7, self.nTimeslot - 0.3, color='k', linewidth=1)
            plt.ylim([0, max(sum(
                np.vstack((P_gen_pri + P_gen_reg + P_gen_sec, P_pump_pri + P_pump_reg + P_pump_sec)))) * 1.1])

            plt.subplot(414)
            if total_flag:
                sumP_gen_pri = sum(P_gen_pri)
                sumP_gen_reg = sum(P_gen_reg)
                sumP_gen_sec = sum(P_gen_sec)
                sumP_gen_ter = sum(P_gen_ter)
                plt.bar(np.arange(self.nTimeslot), sumP_gen_pri + sumP_gen_reg + sumP_gen_sec + sumP_gen_ter, label="Gen_sum")
            else:
                for i in range(self.nGen):
                    plt.bar(np.arange(self.nTimeslot),
                            P_gen_pri[i, :] + P_gen_reg[i, :] + P_gen_sec[i, :] + P_gen_ter[i, :],
                            bottom=sum(P_gen_pri[i, :] + P_gen_reg[i, :] + P_gen_sec[i, :] + P_gen_ter[i, :] for i in
                                       range(i)),
                            label=self.generatorList[i].name)
            if totalPump_Flag:
                sumP_pump_pri = sum(P_pump_pri)
                sumP_pump_reg = sum(P_pump_reg)
                sumP_pump_sec = sum(P_pump_sec)
                sumP_pump_ter = sum(P_pump_terSpin + P_pump_terNspin)
                plt.bar(np.arange(self.nTimeslot), sumP_pump_pri + sumP_pump_reg + sumP_pump_sec + sumP_pump_ter,
                        label="PSH_sum",
                        bottom=sum(P_gen_pri + P_gen_reg + P_gen_sec + P_gen_ter))
            else:
                for j in range(self.nPump):
                    plt.bar(np.arange(self.nTimeslot),
                            P_pump_pri[j, :] + P_pump_reg[j, :] + P_pump_sec[j, :] + P_pump_terSpin[j, :] + P_pump_terNspin[j, :],
                            bottom=sum(P_pump_pri[j, :] + P_pump_reg[j, :] + P_pump_sec[j, :] + P_pump_terSpin[j, :] + P_pump_terNspin[j, :]
                                       for j in range(j)) + sum(P_gen_pri + P_gen_reg + P_gen_sec + P_gen_ter),
                            label=self.pumpList[j].name)
            plt.legend(loc='best', ncol=3, fontsize=legend_fontsize)
            plt.grid(alpha=0.4, linestyle='--')
            plt.xticks(np.arange(0, self.nTimeslot, 1 / self.UNIT_TIME), fontsize=8)
            plt.ylabel('1차+주∙제+2차+3차 (MW)')
            plt.xlim([-0.7, self.nTimeslot - 0.3])
            plt.axhline(req_pri, -0.7, self.nTimeslot - 0.3, color='k', linewidth=1)
            plt.axhline(req_pri + req_reg, -0.7, self.nTimeslot - 0.3, color='k', linewidth=1)
            plt.axhline(req_pri + req_reg + req_sec, -0.7, self.nTimeslot - 0.3, color='k', linewidth=1)
            plt.axhline(req_pri + req_reg + req_sec + req_ter, -0.7, self.nTimeslot - 0.3, color='k', linewidth=1)
            plt.ylim([0, max(sum(np.vstack((P_gen_pri + P_gen_reg + P_gen_sec + P_gen_ter,
                                            P_pump_pri + P_pump_reg + P_pump_sec + P_pump_terSpin + P_pump_terNspin)))) * 1.1])
            plt.suptitle('Reserve cascading', y=0.91, fontsize=14)
        else:
            plt.rcParams["figure.figsize"] = (6, 10)
            plt.figure(3)
            plt.subplot(511)
            if total_flag:
                sumP_gen_pri = sum(P_gen_pri)
                plt.bar(np.arange(self.nTimeslot), sumP_gen_pri, label="Gen_sum")
            else:
                for i in range(self.nGen):
                    plt.bar(np.arange(self.nTimeslot), P_gen_pri[i, :], bottom=sum(P_gen_pri[i, :] for i in range(i)),
                            label=self.generatorList[i].name)
            if totalPump_Flag:
                sumP_pump_pri = sum(P_pump_pri)
                plt.bar(np.arange(self.nTimeslot), sumP_pump_pri, label="PSH_sum", bottom=sum(P_gen_pri))
            else:
                for j in range(self.nPump):
                    plt.bar(np.arange(self.nTimeslot), P_pump_pri[j, :], bottom=sum(P_pump_pri[j, :] for j in range(j))+sum(P_gen_pri),
                            label=self.pumpList[j].name)
            plt.legend(loc='best', ncol=3, fontsize=legend_fontsize)
            plt.grid(alpha=0.4, linestyle='--')
            plt.xticks(np.arange(0, self.nTimeslot, 1 / self.UNIT_TIME), fontsize=8)
            plt.ylabel('1차예비력 (MW)')
            plt.xlim([-0.7, self.nTimeslot - 0.3])
            plt.axhline(req_pri, -0.7, self.nTimeslot - 0.3, color='k', linewidth=1)
            plt.ylim([0, max(sum(np.vstack((P_gen_pri, P_pump_pri))))*1.1])

            plt.subplot(512)
            if total_flag:
                sumP_gen_reg = sum(P_gen_reg)
                plt.bar(np.arange(self.nTimeslot), sumP_gen_reg, label="Gen_sum")
            else:
                for i in range(self.nGen):
                    plt.bar(np.arange(self.nTimeslot), P_gen_reg[i, :], bottom=sum(P_gen_reg[i, :] for i in range(i)),
                            label=self.generatorList[i].name)
            if totalPump_Flag:
                sumP_pump_reg = sum(P_pump_reg)
                plt.bar(np.arange(self.nTimeslot), sumP_pump_reg, label="PSH_sum", bottom=sum(P_gen_reg))
            else:
                for j in range(self.nPump):
                    plt.bar(np.arange(self.nTimeslot), P_pump_reg[j, :], bottom=sum(P_pump_reg[j, :] for j in range(j))+sum(P_gen_reg),
                            label=self.pumpList[j].name)
            plt.legend(loc='best', ncol=3, fontsize=legend_fontsize)
            plt.grid(alpha=0.4, linestyle='--')
            plt.xticks(np.arange(0, self.nTimeslot, 1 / self.UNIT_TIME), fontsize=8)
            plt.ylabel('주파수제어예비력 (MW)')
            plt.xlim([-0.7, self.nTimeslot - 0.3])
            plt.axhline(req_reg, -0.7, self.nTimeslot - 0.3, color='k', linewidth=1)
            plt.ylim([0, max(sum(np.vstack((P_gen_reg, P_pump_reg))))*1.1])

            plt.subplot(513)
            if total_flag:
                sumP_gen_reg = sum(P_gen_reg)
                sumP_gen_sec = sum(P_gen_sec)
                plt.bar(np.arange(self.nTimeslot), sumP_gen_reg + sumP_gen_sec, label="Gen_sum")
            else:
                for i in range(self.nGen):
                    plt.bar(np.arange(self.nTimeslot), P_gen_reg[i, :] + P_gen_sec[i, :],
                            bottom=sum(P_gen_reg[i, :] + P_gen_sec[i, :] for i in range(i)),
                            label=self.generatorList[i].name)
            if totalPump_Flag:
                sumP_pump_reg = sum(P_pump_reg)
                sumP_pump_sec = sum(P_pump_sec)
                plt.bar(np.arange(self.nTimeslot), sumP_pump_reg + sumP_pump_sec, label="PSH_sum",
                        bottom=sum(P_gen_reg+P_gen_sec))
            else:
                for j in range(self.nPump):
                    plt.bar(np.arange(self.nTimeslot), P_pump_reg[j, :] + P_pump_sec[j, :],
                            bottom=sum(P_pump_reg[j, :] + P_pump_sec[j, :] for j in range(j))+sum(P_gen_reg + P_gen_sec),
                            label=self.pumpList[j].name)
            plt.legend(loc='best', ncol=3, fontsize=legend_fontsize)
            plt.grid(alpha=0.4, linestyle='--')
            plt.xticks(np.arange(0, self.nTimeslot, 1 / self.UNIT_TIME), fontsize=8)
            plt.ylabel('주∙제+2차 (MW)')
            plt.xlim([-0.7, self.nTimeslot - 0.3])
            plt.axhline(req_reg, -0.7, self.nTimeslot - 0.3, color='k', linewidth=1)
            plt.axhline(req_reg+req_sec, -0.7, self.nTimeslot - 0.3, color='k', linewidth=1)
            plt.ylim([0, max(sum(np.vstack((P_gen_reg+P_gen_sec, P_pump_reg+P_pump_sec))))*1.1])

            plt.subplot(514)
            if total_flag:
                sumP_gen_reg = sum(P_gen_reg)
                sumP_gen_sec = sum(P_gen_sec)
                sumP_gen_ter = sum(P_gen_ter)
                plt.bar(np.arange(self.nTimeslot), sumP_gen_reg + sumP_gen_sec + sumP_gen_ter, label="Gen_sum")
            else:
                for i in range(self.nGen):
                    plt.bar(np.arange(self.nTimeslot), P_gen_reg[i, :]+P_gen_sec[i, :]+P_gen_ter[i, :],
                            bottom=sum(P_gen_reg[i, :]+P_gen_sec[i, :]+P_gen_ter[i, :] for i in range(i)),
                            label=self.generatorList[i].name)
            if totalPump_Flag:
                sumP_pump_reg = sum(P_pump_reg)
                sumP_pump_sec = sum(P_pump_sec)
                sumP_pump_ter = sum(P_pump_terSpin + P_pump_terNspin)
                plt.bar(np.arange(self.nTimeslot), sumP_pump_reg + sumP_pump_sec + sumP_pump_ter, label="PSH_sum",
                        bottom=sum(P_gen_reg+P_gen_sec+P_gen_ter))
            else:
                for j in range(self.nPump):
                    plt.bar(np.arange(self.nTimeslot), P_pump_reg[j, :]+P_pump_sec[j, :]+P_pump_terSpin[j, :] + P_pump_terNspin[j, :],
                            bottom=sum(P_pump_reg[j, :]+P_pump_sec[j, :]+P_pump_terSpin[j, :] + P_pump_terNspin[j, :]
                                       for j in range(j))+sum(P_gen_reg+P_gen_sec+P_gen_ter),
                            label=self.pumpList[j].name)
            plt.legend(loc='best', ncol=3, fontsize=legend_fontsize)
            plt.grid(alpha=0.4, linestyle='--')
            plt.xticks(np.arange(0, self.nTimeslot, 1 / self.UNIT_TIME), fontsize=8)
            plt.ylabel('주∙제+2차+3차 (MW)')
            plt.xlim([-0.7, self.nTimeslot - 0.3])
            plt.axhline(req_reg, -0.7, self.nTimeslot - 0.3, color='k', linewidth=1)
            plt.axhline(req_reg + req_sec, -0.7, self.nTimeslot - 0.3, color='k', linewidth=1)
            plt.axhline(req_reg + req_sec + req_ter, -0.7, self.nTimeslot - 0.3, color='k', linewidth=1)
            plt.ylim([0, max(sum(np.vstack((P_gen_reg+P_gen_sec+P_gen_ter, P_pump_reg+P_pump_sec+P_pump_terSpin+P_pump_terNspin))))*1.1])

            plt.subplot(515)
            if total_flag:
                sumP_gen_pri = sum(P_gen_pri)
                sumP_gen_reg = sum(P_gen_reg)
                sumP_gen_sec = sum(P_gen_sec)
                sumP_gen_ter = sum(P_gen_ter)
                plt.bar(np.arange(self.nTimeslot), sumP_gen_pri + sumP_gen_reg + sumP_gen_sec + sumP_gen_ter, label="Gen_sum")
            else:
                for i in range(self.nGen):
                    plt.bar(np.arange(self.nTimeslot), P_gen_pri[i, :]+P_gen_reg[i, :]+P_gen_sec[i, :]+P_gen_ter[i, :],
                            bottom=sum(P_gen_pri[i, :]+P_gen_reg[i, :]+P_gen_sec[i, :]+P_gen_ter[i, :] for i in range(i)),
                            label=self.generatorList[i].name)
            if totalPump_Flag:
                sumP_pump_pri = sum(P_pump_pri)
                sumP_pump_reg = sum(P_pump_reg)
                sumP_pump_sec = sum(P_pump_sec)
                sumP_pump_ter = sum(P_pump_terSpin + P_pump_terNspin)
                plt.bar(np.arange(self.nTimeslot), sumP_pump_pri + sumP_pump_reg + sumP_pump_sec + sumP_pump_ter, label="PSH_sum",
                        bottom=sum(P_gen_pri+P_gen_reg+P_gen_sec+P_gen_ter))
            else:
                for j in range(self.nPump):
                    plt.bar(np.arange(self.nTimeslot), P_pump_pri[j, :]+P_pump_reg[j, :]+P_pump_sec[j, :]+P_pump_terSpin[j, :] + P_pump_terNspin[j, :],
                            bottom=sum(P_pump_pri[j, :]+P_pump_reg[j, :]+P_pump_sec[j, :]+P_pump_terSpin[j, :] + P_pump_terNspin[j, :]
                                       for j in range(j))+sum(P_gen_pri+P_gen_reg+P_gen_sec+P_gen_ter),
                            label=self.pumpList[j].name)
            plt.legend(loc='best', ncol=3, fontsize=legend_fontsize)
            plt.grid(alpha=0.4, linestyle='--')
            plt.xticks(np.arange(0, self.nTimeslot, 1 / self.UNIT_TIME), fontsize=8)
            plt.ylabel('1차+주∙제+2차+3차 (MW)')
            plt.xlim([-0.7, self.nTimeslot - 0.3])
            plt.axhline(req_pri, -0.7, self.nTimeslot - 0.3, color='k', linewidth=1)
            plt.axhline(req_pri+req_reg, -0.7, self.nTimeslot - 0.3, color='k', linewidth=1)
            plt.axhline(req_pri+req_reg + req_sec, -0.7, self.nTimeslot - 0.3, color='k', linewidth=1)
            plt.axhline(req_pri+req_reg + req_sec + req_ter, -0.7, self.nTimeslot - 0.3, color='k', linewidth=1)
            plt.ylim([0, max(sum(np.vstack((P_gen_pri+P_gen_reg+P_gen_sec+P_gen_ter, P_pump_pri+P_pump_reg+P_pump_sec+P_pump_terSpin+P_pump_terNspin))))*1.1])
            plt.suptitle('Reserve cascading', y=0.91, fontsize=14)
        if save_flag:
            plt.savefig('./'+time.strftime("%m%d%H%M")+'_cascading.png', dpi=300, bbox_inches='tight')
        if last_flag:
            plt.show()
