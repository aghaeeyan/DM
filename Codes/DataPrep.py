import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import copy

dataFile1 = 'rawData/cases_pt.csv' #https://github.com/ccodwg/CovidTimelineCanada/tree/main/data/pt
dataFile1_1 = 'rawData/deaths_pt.csv' #https://github.com/ccodwg/CovidTimelineCanada/tree/main/data/pt
dataFile2_1 = 'rawData/vaccine_administration_dose_1_pt.csv'#https://github.com/ccodwg/CovidTimelineCanada/tree/main/data/pt
dataFile2_2 = 'rawData/vaccine_administration_dose_2_pt.csv' #https://github.com/ccodwg/CovidTimelineCanada/tree/main/data/pt
dataFile2_3 = 'rawData/vaccine_administration_dose_3_pt.csv' #https://github.com/ccodwg/CovidTimelineCanada/tree/main/data/pt
dataFile2_4 = 'rawData/vaccine_administration_dose_4_pt.csv' #https://github.com/ccodwg/CovidTimelineCanada/tree/main/data/pt
dataFile3 = 'Data/popCan.csv' #https://www150.statcan.gc.ca/t1/tbl1/en/tv.action?pid=1710000501
DistFile = 'vaccination-distribution.csv' #https://healthinfobase.canada.ca/covid-19/vaccine-distribution/
Vax2024File = 'vaccination-coverage-map.csv' #https://health-infobase.canada.ca/covid-19/vaccination-coverage/
nonhesitant = pd.read_csv('nonhesitant.csv') # June 18, 2023 https://health-infobase.canada.ca/covid-19/vaccination-coverage/
RecentVax = pd.read_csv(Vax2024File)
RecentVax['week_end'] = pd.to_datetime(RecentVax['week_end'])
RecentVax = RecentVax[['week_end', 'prename', 'num12plus_atleast1dose']]
non = pd.DataFrame(columns=['Location', 'nonhesitant'])
Dist = pd.read_csv(DistFile) # vaccine distribution
popf = pd.read_csv(dataFile3) #population of Canada
CaseData = pd.read_csv(dataFile1)
DeathData = pd.read_csv(dataFile1_1)
FirstVac = pd.read_csv(dataFile2_1)
SecVac = pd.read_csv(dataFile2_2)
ThirdVac = pd.read_csv(dataFile2_3)
FourthVac = pd.read_csv(dataFile2_4)
FirstVac = FirstVac.rename(columns={'value':'dose1'})
SecVac = SecVac.rename(columns={'value':'dose2'})
ThirdVac = ThirdVac.rename(columns={'value':'dose3'})
FourthVac = FourthVac.rename(columns={'value':'dose4'})
FirstVac['date'] = pd.to_datetime(FirstVac['date'])
SecVac['date'] = pd.to_datetime(SecVac['date'])
ThirdVac['date'] = pd.to_datetime(ThirdVac['date'])
FourthVac['date'] = pd.to_datetime(FourthVac['date'])
CaseData['date'] = pd.to_datetime(CaseData['date'])
DeathData['date'] = pd.to_datetime(DeathData['date'])
FirstVac.sort_values(by='date', inplace=True)
SecVac.sort_values(by='date', inplace=True)
ThirdVac.sort_values(by='date', inplace=True)
FourthVac.sort_values(by='date', inplace=True)
CaseData.sort_values(by='date', inplace=True)
DeathData.sort_values(by='date', inplace=True)
ProvincesNames = list(FirstVac['region'].unique())
finalFile = pd.DataFrame(columns=['region', 'date', 'case_daily', 'death_daily', 'dose1', 'dose_distributed', 'popf12',
                                  'poptotal'])
FirstSeclVac = pd.merge(SecVac, FirstVac, how='outer', on =['region','date'])
VacUnified_0 = pd.merge(FirstSeclVac, ThirdVac, how='outer', on =['region','date'])
VacUnified = pd.merge(VacUnified_0, FourthVac, how='outer', on =['region','date'])
VacUnified.fillna(0, inplace=True)
VacUnified = VacUnified[VacUnified['dose1'] > 0]
EpiUnified = pd.merge(CaseData, DeathData, on =['region','date'])
EpiUnified.dropna(inplace=True)
EpiVacUnified = pd.merge(EpiUnified, VacUnified, on=['region', 'date'])
# To merge Eligible Population and EpiVac file
Dist_0 = pd.DataFrame(columns=['prename', 'report_date', 'numtotal_all_distributed','numtotal_janssen_distributed','population','Ne'])
Dist_0 = pd.concat([Dist_0,Dist[['prename', 'report_date', 'numtotal_all_distributed','numtotal_janssen_distributed']]])
Dist_0 = Dist_0.rename(columns={'report_date': 'date', 'prename':'region'})
Dist_0 = Dist_0.replace(['Alberta', 'Quebec', 'British Columbia', 'Manitoba', 'New Brunswick', 'Newfoundland and Labrador',
                         'Northwest Territories', 'Nova Scotia', 'Nunavut', 'Prince Edward Island', 'Saskatchewan',
                         'Yukon','Ontario'],
                              ['AB', 'QC', 'BC', 'MB', 'NB', 'NL', 'NT', 'NS', 'NU', 'PE', 'SK', 'YT', 'ON'])

Dist_0['date'] = pd.to_datetime(Dist_0['date'])
for i in ProvincesNames:
    p_all = popf.loc[popf['region'] == i, 'poptotal'].to_numpy(dtype=np.float64)[0]
    p_f12 = popf.loc[popf['region'] == i, 'popf12'].to_numpy(dtype=np.float64)[0]
    p_f18 = popf.loc[popf['region'] == i, 'popf18'].to_numpy(dtype=np.float64)[0]
    Dist_0.loc[(Dist_0['region'] == i), 'poptotal'] = p_all
    Dist_0.loc[(Dist_0['region'] == i), 'popf12'] = p_f12
    nonh = nonhesitant.loc[nonhesitant['Location'] == i, 'coverage'].to_numpy(dtype=np.float64)[0]
    Dist_0.loc[(Dist_0['region'] == i), 'Ne'] = p_f12*nonh #nonh?
EpiVacPop = pd.merge(EpiVacUnified, Dist_0, how='outer', on =['region', 'date'])
EpiVacPop['error_adj'] = EpiVacPop['dose2']- EpiVacPop['numtotal_janssen_distributed']
EpiVacPop.sort_values(by='date', inplace=True)
EpiVacPop.case_daily=EpiVacPop.case_daily.mask(EpiVacPop.case_daily.lt(0), 0)
EpiVacPop.death_daily=EpiVacPop.death_daily.mask(EpiVacPop.death_daily.lt(0), 0)
EpiVacPop['dose_distributed'] = EpiVacPop['numtotal_all_distributed'] - EpiVacPop['error_adj']-\
                       EpiVacPop['dose3'] - EpiVacPop['dose4']
EpiVacPop.sort_values(by='date', inplace=True)
EpiVacPop = EpiVacPop[EpiVacPop['case_daily'].notna()]
EpiVacPop = EpiVacPop[EpiVacPop['death_daily'].notna()]
EpiVacPop = EpiVacPop[EpiVacPop['dose1'].notna()]
EpiVacPop = EpiVacPop[EpiVacPop['dose1'] > 0]
EpiVacPop = EpiVacPop[EpiVacPop['dose_distributed'].notna()]
EpiVacPop.dose_distributed = EpiVacPop.dose_distributed.mask(EpiVacPop.dose_distributed.lt(0),0)
EpiVacPop.dose1 = EpiVacPop.dose1.mask(EpiVacPop.dose1.lt(0),0)
EpiVacPopUnified = copy.deepcopy(EpiVacPop)
# EpiVacPopUnified = pd.merge(EpiVacPop, cutDate, on=['region', 'date'])
namee = []
for stateName in list(EpiVacPopUnified['region'].unique()):
    ffllg = 0
    jadmin = list(EpiVacPopUnified.loc[EpiVacPopUnified['region']==stateName, 'dose_distributed'].to_numpy(dtype=np.float64))
    for jj in range(1,len(jadmin)):
        if jadmin[jj] < jadmin[jj-1]:
            jadmin[jj] = jadmin[jj-1]
            if ffllg < 1:
                ffllg = 1.5

    EpiVacPopUnified.loc[EpiVacPopUnified['region'] == stateName, 'dose_distributed'] = jadmin
finalFile = pd.concat([finalFile,EpiVacPopUnified[['region','date','case_daily','death_daily','dose1','dose_distributed'
    ,'poptotal', 'Ne', 'popf12']]])
finalFile['date'] = pd.to_datetime(finalFile['date'])
finalFile.sort_values(by='date', inplace=True)
finalFile = finalFile.rename(columns={'region':'Location','date':'Date','case_daily':'new_case',
                                      'death_daily':'new_death','dose1':'Admin_Dose_1'
    })
#We did not have access to data on the number of doses allocated to adult only or children. We hence set the last
#date to be the time where children were allowed to receive a dose of COVID-19 vax
#https://www.canada.ca/en/health-canada/news/2021/11/health-canada-authorizes-use-of-comirnaty-the-pfizer-biontech-covid-19-vaccine-in-children-5-to-11-years-of-age.html
finalDate = pd.to_datetime('21/11/2021')
finalFile = finalFile[finalFile.Date < finalDate]
# to adjust the number of vaccinated in the provinces of Nova Scotia and Quebec
for pr in ['NS', 'QC']:
    if pr == 'NS':
       adjusting_coef = RecentVax.loc[(RecentVax['prename'] == 'Nova Scotia') &
                                      (RecentVax['week_end'] == pd.to_datetime('4/23/2023')),
                                      'num12plus_atleast1dose'].values[0]/RecentVax.loc[(RecentVax['prename'] == 'Nova Scotia') &
                                      (RecentVax['week_end'] == pd.to_datetime('3/26/2023')),
                                      'num12plus_atleast1dose'].values[0]
    else:
        adjusting_coef = RecentVax.loc[(RecentVax['prename'] == 'Quebec') &
                                       (RecentVax['week_end'] == pd.to_datetime('8/14/2022')),
                                       'num12plus_atleast1dose'].values[0] / \
                         RecentVax.loc[(RecentVax['prename'] == 'Quebec') &
                                       (RecentVax['week_end'] == pd.to_datetime('7/17/2022')),
                                       'num12plus_atleast1dose'].values[0]
    finalFile.loc[finalFile['Location'] == pr, 'Admin_Dose_1'] = \
        finalFile.loc[finalFile['Location'] == pr, 'Admin_Dose_1']*adjusting_coef
#Converting daily data to weekly
logic = {'Admin_Dose_1':'last','dose_distributed':'last','new_case':'sum','new_death':'sum','poptotal':'last',
         'Ne':'last', 'popf12':'last'}
dataf_w = pd.DataFrame(columns=['Location', 'Date', 'new_case', 'new_death','Admin_Dose_1','dose_distributed',
                                'popf12','Ne', 'poptotal'])


for state in ProvincesNames:
    dataf = pd.DataFrame(columns=['new_case', 'new_death', 'Admin_Dose_1', 'dose_distributed', 'poptotal', 'Ne', 'popf12'])
    f_state = finalFile.loc[finalFile['Location'] == state]
    f_state=f_state.resample('W')\
    .apply(logic)
    dataf= pd.concat([f_state, dataf])
    dataf['Location'] = state
    dataf_w = pd.concat([dataf_w, dataf])
filepath1 = Path('Data/EpiVacFile_6_1_week.csv')
filepath1.parent.mkdir(parents=True, exist_ok=True)
dataf_w.to_csv(filepath1)





