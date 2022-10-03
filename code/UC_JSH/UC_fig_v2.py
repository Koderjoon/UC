import numpy as np
import matplotlib.pyplot as plt

class UC_plot:
    def __init__(self, UNIT_TIME, generatorList, nuclearList, pumpList, essList, load):
        self.nTimeslot = len(load)
        self.UNIT_TIME = UNIT_TIME
        self.generatorList = generatorList
        self.nuclearList = nuclearList
        self.pumpList = pumpList
        self.essList = essList
        self.nGen = len(generatorList)
        self.nNuc = len(nuclearList)
        self.nPump = len(pumpList)
        self.nEss = len(essList)
        self.load = load

    def make_plot(self, P_sol, U_sol, SoC_sol, save_flag=False):
        [P_genSol, P_nuclearSol, P_pumpDisSol, P_pumpChgSol, P_essDisSol, P_essChgSol] = P_sol
        [U_genSol, SU_genSol, SD_genSol, U_nuclearSol, SU_nuclearSol, SD_nuclearSol, U_pumpDisSol, U_pumpChgSol, U_essDisSol, U_essChgSol] = U_sol
        [socPump, socESS] = SoC_sol

        plt.rcParams["figure.figsize"] = (6, 6)
        plt.figure(1)
        plt.subplot(211)
        plt.plot(np.arange(self.nTimeslot), self.load, label='Load')
        for i in range(self.nGen):
            plt.plot(np.arange(self.nTimeslot), P_genSol[i, :], label='P_gen_' + str(i + 1))
        for i in range(self.nPump):
            plt.plot(np.arange(self.nTimeslot), P_pumpDisSol[i, :], label='P_pumpDis_' + str(i + 1))
            plt.plot(np.arange(self.nTimeslot), -P_pumpChgSol[i, :], label='P_pumpChg_' + str(i + 1))
        for i in range(self.nEss):
            plt.plot(np.arange(self.nTimeslot), P_essDisSol[i, :], label='P_essDis_' + str(i + 1))
            plt.plot(np.arange(self.nTimeslot), -P_essChgSol[i, :], label='P_essChg_' + str(i + 1))
        plt.legend(loc='best', ncol=3, fontsize='x-small')
        plt.grid()
        plt.xticks(np.arange(0, self.nTimeslot, 1 / self.UNIT_TIME), fontsize=8)
        plt.ylabel('Power (MW)')
        plt.xlim([0, self.nTimeslot - 1])

        plt.subplot(212)
        for i in range(self.nEss):
            plt.plot(np.arange(self.nTimeslot), socESS[i, :], label='SoC_ESS' + str(i + 1))
        for j in range(self.nPump):
            plt.plot(np.arange(self.nTimeslot), socPump[j, :], label='SoC_pump' + str(j + 1))
        plt.legend(loc='best', ncol=2, fontsize='x-small')
        plt.grid()
        plt.xticks(np.arange(0, self.nTimeslot, 1 / self.UNIT_TIME), fontsize=8)
        plt.ylim([0, 1])
        plt.ylabel('SoC')
        plt.xlim([0, self.nTimeslot - 1])
        if save_flag:
            plt.savefig('./test_total.png', dpi=300, bbox_inches='tight')
        # plt.show()

        plt.rcParams["figure.figsize"] = (6, 6)
        plt.figure(2)
        plt.subplot(411)
        for i in range(self.nGen):
            plt.plot(np.arange(self.nTimeslot), U_genSol[i, :], label='U_gen_' + str(i + 1))
        plt.legend(loc='upper right', ncol=1, fontsize='x-small')
        plt.grid()
        plt.xticks(np.arange(0, self.nTimeslot, 4 / self.UNIT_TIME), fontsize=8)
        plt.ylim([-0.1, 1.1])
        plt.xlim([0, self.nTimeslot - 1])
        plt.ylabel('Binary')
        plt.subplot(412)
        for i in range(self.nGen):
            plt.plot(np.arange(self.nTimeslot), SU_genSol[i, :], label='SU_' + str(i + 1))
            plt.plot(np.arange(self.nTimeslot), SD_genSol[i, :], label='SD_' + str(i + 1))
        plt.legend(loc='upper right', ncol=2, fontsize='x-small')
        plt.grid()
        plt.xticks(np.arange(0, self.nTimeslot, 4 / self.UNIT_TIME), fontsize=8)
        plt.ylim([-0.1, 1.1])
        plt.xlim([0, self.nTimeslot - 1])
        plt.ylabel('Binary')
        plt.subplot(413)
        for i in range(self.nPump):
            plt.plot(np.arange(self.nTimeslot), U_pumpDisSol[i, :], label='U_pumpDis_' + str(i + 1))
            plt.plot(np.arange(self.nTimeslot), U_pumpChgSol[i, :], label='U_pumpChg_' + str(i + 1))
        plt.legend(loc='upper right', ncol=2, fontsize='x-small')
        plt.grid()
        plt.xticks(np.arange(0, self.nTimeslot, 4 / self.UNIT_TIME), fontsize=8)
        plt.ylim([-0.1, 1.1])
        plt.xlim([0, self.nTimeslot - 1])
        plt.ylabel('Binary')
        plt.subplot(414)
        for i in range(self.nEss):
            plt.plot(np.arange(self.nTimeslot), U_essDisSol[i, :], label='U_essDis_' + str(i + 1))
            plt.plot(np.arange(self.nTimeslot), U_essChgSol[i, :], label='U_essChg_' + str(i + 1))
        plt.legend(loc='upper right', ncol=2, fontsize='x-small')
        plt.grid()
        plt.ylim([-0.1, 1.1])
        plt.xlim([0, self.nTimeslot - 1])
        plt.xticks(np.arange(0, self.nTimeslot, 4 / self.UNIT_TIME), fontsize=8)
        plt.ylabel('Binary')
        if save_flag:
            plt.savefig('./test_binary.png', dpi=300, bbox_inches='tight')
        plt.show()

    def make_plot_new(self, P_sol, SoC_sol, total_flag=False, save_flag=False, last_flag=False):
        [P_genSol, P_nuclearSol, P_pumpDisSol, P_pumpChgSol, P_essDisSol, P_essChgSol] = P_sol
        [socPump, socESS] = SoC_sol

        plt.rcParams["figure.figsize"] = (6, 9)
        plt.rcParams['font.family'] = 'Malgun Gothic'
        plt.rcParams['axes.unicode_minus'] = False

        plt.figure()
        plt.subplot(311)
        plt.plot(np.arange(self.nTimeslot), self.load, label='Load', color='navy')
        
        if total_flag:
            plt.plot(np.arange(self.nTimeslot), sum(P_genSol), label='gen')
        else:
            for i in range(self.nGen):
                plt.plot(np.arange(self.nTimeslot), P_genSol[i, :], label=self.generatorList[i].name)
            for i in range(self.nNuc):
                plt.plot(np.arange(self.nTimeslot), P_nuclearSol[i, :], label=self.nuclearList[i].name,
                         color='C' + str(i + self.nGen + self.nPump + self.nEss))
        plt.legend(loc='best', ncol=2, fontsize='x-small')
        plt.grid(alpha=0.4, linestyle='--')
        plt.xticks(np.arange(0, self.nTimeslot, 1 / self.UNIT_TIME), fontsize=8)
        plt.ylabel('Load & Gen (MW)')
        plt.xlim([0, self.nTimeslot - 1])

        plt.subplot(312)
        for i in range(self.nPump):
            plt.plot(np.arange(self.nTimeslot), P_pumpDisSol[i, :] - P_pumpChgSol[i, :], label=self.pumpList[i].name,
                     color='C' + str(i + self.nGen))
        for i in range(self.nEss):
            plt.plot(np.arange(self.nTimeslot), P_essDisSol[i, :] - P_essChgSol[i, :], label=self.essList[i].name,
                     color='C' + str(i + self.nGen + self.nPump))
        plt.legend(loc='best', ncol=2, fontsize='x-small')
        plt.grid(alpha=0.4, linestyle='--')
        plt.xticks(np.arange(0, self.nTimeslot, 1 / self.UNIT_TIME), fontsize=8)
        plt.ylabel('Pump & ESS (MW)')
        plt.xlim([0, self.nTimeslot - 1])

        plt.subplot(313)
        for i in range(self.nPump):
            plt.plot(np.arange(self.nTimeslot), socPump[i, :], label=self.pumpList[i].name,
                     color='C' + str(i + self.nGen))
        for i in range(self.nEss):
            plt.plot(np.arange(self.nTimeslot), socESS[i, :], label=self.essList[i].name,
                     color='C' + str(i + self.nGen + self.nPump))
        plt.legend(loc='best', ncol=2, fontsize='x-small')
        plt.grid(alpha=0.4, linestyle='--')
        plt.xticks(np.arange(0, self.nTimeslot, 1 / self.UNIT_TIME), fontsize=8)
        plt.ylim([0, 1])
        plt.ylabel('SoC')
        plt.xlim([0, self.nTimeslot - 1])
        plt.suptitle('UC results', y=0.91, fontsize=14)
        if save_flag:
            plt.savefig('./test_total.png', dpi=300, bbox_inches='tight')
        if last_flag:
            plt.show()

    def make_res_plot(self, GF_sol, AGC_FC_sol, AGC_sec_sol, Spin_sol, Nspin_sol, last_flag=False, save_flag=False):
        [P_genGF, P_pumpGF] = GF_sol
        [P_genAGC_FC, P_pumpAGC_FC] = AGC_FC_sol
        [P_genAGC_sec, P_pumpAGC_sec] = AGC_sec_sol
        [P_genSpin, P_pumpSpin] = Spin_sol
        [P_genNspin, P_pumpNspin] = Nspin_sol

        plt.rcParams["figure.figsize"] = (6, 9)
        plt.rcParams['font.family'] = 'Malgun Gothic'
        plt.rcParams['axes.unicode_minus'] = False
        legend_fontsize = 7

        plt.figure()
        plt.subplot(511)
        for i in range(self.nGen):
            plt.bar(np.arange(self.nTimeslot), P_genGF[i, :], bottom=sum(P_genGF[i, :] for i in range(i)),
                    label=self.generatorList[i].name)
        for j in range(self.nPump):
            plt.bar(np.arange(self.nTimeslot), P_pumpGF[j, :], bottom=sum(P_pumpGF[j, :] for j in range(j))+sum(P_genGF),
                    label=self.pumpList[j].name)
        plt.legend(loc='best', ncol=2, fontsize=legend_fontsize)
        plt.grid(alpha=0.4, linestyle='--')
        plt.xticks(np.arange(0, self.nTimeslot, 1 / self.UNIT_TIME), fontsize=8)
        plt.ylabel('1차예비력 (MW)')
        plt.xlim([-0.7, self.nTimeslot - 0.3])
        plt.ylim([0, max(sum(np.vstack((P_genGF, P_pumpGF))))*1.1])

        plt.subplot(512)
        for i in range(self.nGen):
            plt.bar(np.arange(self.nTimeslot), P_genAGC_FC[i, :], bottom=sum(P_genAGC_FC[i, :] for i in range(i)),
                    label=self.generatorList[i].name)
        for j in range(self.nPump):
            plt.bar(np.arange(self.nTimeslot), P_pumpAGC_FC[j, :], bottom=sum(P_pumpAGC_FC[j, :] for j in range(j))+sum(P_genAGC_FC),
                    label=self.pumpList[j].name)
        plt.legend(loc='best', ncol=2, fontsize=legend_fontsize)
        plt.grid(alpha=0.4, linestyle='--')
        plt.xticks(np.arange(0, self.nTimeslot, 1 / self.UNIT_TIME), fontsize=8)
        plt.ylabel('주파수제어예비력 (MW)')
        plt.xlim([-0.7, self.nTimeslot - 0.3])
        plt.ylim([0, max(sum(np.vstack((P_genAGC_FC, P_pumpAGC_FC)))) * 1.1])

        plt.subplot(513)
        for i in range(self.nGen):
            plt.bar(np.arange(self.nTimeslot), P_genAGC_sec[i, :], bottom=sum(P_genAGC_sec[i, :] for i in range(i)),
                    label=self.generatorList[i].name)
        for j in range(self.nPump):
            plt.bar(np.arange(self.nTimeslot), P_pumpAGC_sec[j, :], bottom=sum(P_pumpAGC_sec[j, :] for j in range(j))+sum(P_genAGC_sec),
                    label=self.pumpList[j].name)
        plt.legend(loc='best', ncol=2, fontsize=legend_fontsize)
        plt.grid(alpha=0.4, linestyle='--')
        plt.xticks(np.arange(0, self.nTimeslot, 1 / self.UNIT_TIME), fontsize=8)
        plt.ylabel('2차예비력 (MW)')
        plt.xlim([-0.7, self.nTimeslot - 0.3])
        plt.ylim([0, max(sum(np.vstack((P_genAGC_sec, P_pumpAGC_sec)))) * 1.1])

        plt.subplot(514)
        for i in range(self.nGen):
            plt.bar(np.arange(self.nTimeslot), P_genSpin[i, :], bottom=sum(P_genSpin[i, :] for i in range(i)),
                    label=self.generatorList[i].name)
        for j in range(self.nPump):
            plt.bar(np.arange(self.nTimeslot), P_pumpSpin[j, :], bottom=sum(P_pumpSpin[j, :] for j in range(j))+sum(P_genSpin),
                    label=self.pumpList[j].name)
        plt.legend(loc='best', ncol=2, fontsize=legend_fontsize)
        plt.grid(alpha=0.4, linestyle='--')
        plt.xticks(np.arange(0, self.nTimeslot, 1 / self.UNIT_TIME), fontsize=8)
        plt.ylabel('3차예비력 (MW)')
        plt.xlim([-0.7, self.nTimeslot - 0.3])
        plt.ylim([0, max(sum(np.vstack((P_genSpin, P_pumpSpin)))) * 1.1])

        plt.subplot(515)
        for i in range(self.nGen):
            plt.bar(np.arange(self.nTimeslot), P_genNspin[i, :], bottom=sum(P_genNspin[i, :] for i in range(i)))
                    #, label=self.generatorList[i].name)
        for j in range(self.nPump):
            plt.bar(np.arange(self.nTimeslot), P_pumpNspin[j, :], bottom=sum(P_pumpNspin[j, :] for j in range(j))+sum(P_genNspin),
                    label=self.pumpList[j].name)
        plt.legend(loc='best', ncol=2, fontsize=legend_fontsize)
        plt.grid(alpha=0.4, linestyle='--')
        plt.xticks(np.arange(0, self.nTimeslot, 1 / self.UNIT_TIME), fontsize=8)
        plt.ylabel('속응성자원 (MW)')
        plt.xlim([-0.7, self.nTimeslot - 0.3])
        plt.ylim([0, max(sum(np.vstack((P_genNspin, P_pumpNspin)))) * 1.1])
        plt.suptitle('Reserve results', y=0.91, fontsize=14)
        if save_flag:
            plt.savefig('./test_reserve.png', dpi=300, bbox_inches='tight')
        if last_flag:
            plt.show()

    def make_res_cascading(self, GF_sol, AGC_FC_sol, AGC_sec_sol, Spin_sol, Nspin_sol, scale, last_flag=False, save_flag=False):
        [P_genGF, P_pumpGF] = GF_sol
        [P_genAGC_FC, P_pumpAGC_FC] = AGC_FC_sol
        [P_genAGC_sec, P_pumpAGC_sec] = AGC_sec_sol
        [P_genSpin, P_pumpSpin] = Spin_sol
        [P_genNspin, P_pumpNspin] = Nspin_sol
        [req_FC, req_GF, req_2, req_3, req_Nspin] = np.array([700, 1000, 1400, 1400, 2000]) / scale

        plt.rcParams["figure.figsize"] = (6, 10)
        plt.rcParams['font.family'] = 'Malgun Gothic'
        plt.rcParams['axes.unicode_minus'] = False
        legend_fontsize = 7

        plt.figure(3)
        plt.subplot(511)
        for i in range(self.nGen):
            plt.bar(np.arange(self.nTimeslot), P_genGF[i, :], bottom=sum(P_genGF[i, :] for i in range(i)),
                    label=self.generatorList[i].name)
        for j in range(self.nPump):
            plt.bar(np.arange(self.nTimeslot), P_pumpGF[j, :], bottom=sum(P_pumpGF[j, :] for j in range(j))+sum(P_genGF),
                    label=self.pumpList[j].name)
        plt.legend(loc='best', ncol=2, fontsize=legend_fontsize)
        plt.grid(alpha=0.4, linestyle='--')
        plt.xticks(np.arange(0, self.nTimeslot, 1 / self.UNIT_TIME), fontsize=8)
        plt.ylabel('1차예비력 (MW)')
        plt.xlim([-0.7, self.nTimeslot - 0.3])
        plt.axhline(req_GF, -0.7, self.nTimeslot - 0.3, color='k', linewidth=1)
        plt.ylim([0, max(sum(np.vstack((P_genGF, P_pumpGF))))*1.1])

        plt.subplot(512)
        for i in range(self.nGen):
            plt.bar(np.arange(self.nTimeslot), P_genAGC_FC[i, :], bottom=sum(P_genAGC_FC[i, :] for i in range(i)),
                    label=self.generatorList[i].name)
        for j in range(self.nPump):
            plt.bar(np.arange(self.nTimeslot), P_pumpAGC_FC[j, :], bottom=sum(P_pumpAGC_FC[j, :] for j in range(j))+sum(P_genAGC_FC),
                    label=self.pumpList[j].name)
        plt.legend(loc='best', ncol=2, fontsize=legend_fontsize)
        plt.grid(alpha=0.4, linestyle='--')
        plt.xticks(np.arange(0, self.nTimeslot, 1 / self.UNIT_TIME), fontsize=8)
        plt.ylabel('주파수제어예비력 (MW)')
        plt.xlim([-0.7, self.nTimeslot - 0.3])
        plt.axhline(req_FC, -0.7, self.nTimeslot - 0.3, color='k', linewidth=1)
        plt.ylim([0, max(sum(np.vstack((P_genAGC_FC, P_pumpAGC_FC))))*1.1])

        plt.subplot(513)
        for i in range(self.nGen):
            plt.bar(np.arange(self.nTimeslot), P_genAGC_FC[i, :] + P_genAGC_sec[i, :],
                    bottom=sum(P_genAGC_FC[i, :] + P_genAGC_sec[i, :] for i in range(i)),
                    label=self.generatorList[i].name)
        for j in range(self.nPump):
            plt.bar(np.arange(self.nTimeslot), P_pumpAGC_FC[j, :] + P_pumpAGC_sec[j, :],
                    bottom=sum(P_pumpAGC_FC[j, :] + P_pumpAGC_sec[j, :] for j in range(j))+sum(P_genAGC_FC + P_genAGC_sec),
                    label=self.pumpList[j].name)
        plt.legend(loc='best', ncol=2, fontsize=legend_fontsize)
        plt.grid(alpha=0.4, linestyle='--')
        plt.xticks(np.arange(0, self.nTimeslot, 1 / self.UNIT_TIME), fontsize=8)
        plt.ylabel('주∙제+2차 (MW)')
        plt.xlim([-0.7, self.nTimeslot - 0.3])
        plt.axhline(req_FC, -0.7, self.nTimeslot - 0.3, color='k', linewidth=1)
        plt.axhline(req_FC+req_2, -0.7, self.nTimeslot - 0.3, color='k', linewidth=1)
        plt.ylim([0, max(sum(np.vstack((P_genAGC_FC+P_genAGC_sec, P_pumpAGC_FC+P_pumpAGC_sec))))*1.1])

        plt.subplot(514)
        for i in range(self.nGen):
            plt.bar(np.arange(self.nTimeslot), P_genAGC_FC[i, :]+P_genAGC_sec[i, :]+P_genSpin[i, :],
                    bottom=sum(P_genAGC_FC[i, :]+P_genAGC_sec[i, :]+P_genSpin[i, :] for i in range(i)),
                    label=self.generatorList[i].name)
        for j in range(self.nPump):
            plt.bar(np.arange(self.nTimeslot), P_pumpAGC_FC[j, :]+P_pumpAGC_sec[j, :]+P_pumpSpin[j, :],
                    bottom=sum(P_pumpAGC_FC[j, :]+P_pumpAGC_sec[j, :]+P_pumpSpin[j, :]
                               for j in range(j))+sum(P_genAGC_FC+P_genAGC_sec+P_genSpin),
                    label=self.pumpList[j].name)
        plt.legend(loc='best', ncol=2, fontsize=legend_fontsize)
        plt.grid(alpha=0.4, linestyle='--')
        plt.xticks(np.arange(0, self.nTimeslot, 1 / self.UNIT_TIME), fontsize=8)
        plt.ylabel('주∙제+2차+3차 (MW)')
        plt.xlim([-0.7, self.nTimeslot - 0.3])
        plt.axhline(req_FC, -0.7, self.nTimeslot - 0.3, color='k', linewidth=1)
        plt.axhline(req_FC + req_2, -0.7, self.nTimeslot - 0.3, color='k', linewidth=1)
        plt.axhline(req_FC + req_2 + req_3, -0.7, self.nTimeslot - 0.3, color='k', linewidth=1)
        plt.ylim([0, max(sum(np.vstack((P_genAGC_FC+P_genAGC_sec+P_genSpin, P_pumpAGC_FC+P_pumpAGC_sec+P_pumpSpin))))*1.1])

        plt.subplot(515)
        for i in range(self.nGen):
            plt.bar(np.arange(self.nTimeslot), P_genGF[i, :]+P_genAGC_FC[i, :]+P_genAGC_sec[i, :]+P_genSpin[i, :],
                    bottom=sum(P_genGF[i, :]+P_genAGC_FC[i, :]+P_genAGC_sec[i, :]+P_genSpin[i, :] for i in range(i)),
                    label=self.generatorList[i].name)
        for j in range(self.nPump):
            plt.bar(np.arange(self.nTimeslot), P_pumpGF[j, :]+P_pumpAGC_FC[j, :]+P_pumpAGC_sec[j, :]+P_pumpSpin[j, :],
                    bottom=sum(P_pumpGF[j, :]+P_pumpAGC_FC[j, :]+P_pumpAGC_sec[j, :]+P_pumpSpin[j, :]
                               for j in range(j))+sum(P_genGF+P_genAGC_FC+P_genAGC_sec+P_genSpin),
                    label=self.pumpList[j].name)
        plt.legend(loc='best', ncol=2, fontsize=legend_fontsize)
        plt.grid(alpha=0.4, linestyle='--')
        plt.xticks(np.arange(0, self.nTimeslot, 1 / self.UNIT_TIME), fontsize=8)
        plt.ylabel('1차+주∙제+2차+3차 (MW)')
        plt.xlim([-0.7, self.nTimeslot - 0.3])
        plt.axhline(req_GF, -0.7, self.nTimeslot - 0.3, color='k', linewidth=1)
        plt.axhline(req_GF+req_FC, -0.7, self.nTimeslot - 0.3, color='k', linewidth=1)
        plt.axhline(req_GF+req_FC + req_2, -0.7, self.nTimeslot - 0.3, color='k', linewidth=1)
        plt.axhline(req_GF+req_FC + req_2 + req_3, -0.7, self.nTimeslot - 0.3, color='k', linewidth=1)
        plt.ylim([0, max(sum(np.vstack((P_genGF+P_genAGC_FC+P_genAGC_sec+P_genSpin, P_pumpGF+P_pumpAGC_FC+P_pumpAGC_sec+P_pumpSpin))))*1.1])
        plt.suptitle('Reserve cascading', y=0.91, fontsize=14)
        if save_flag:
            plt.savefig('./test_res_cascading.png', dpi=300, bbox_inches='tight')
        if last_flag:
            plt.show()
