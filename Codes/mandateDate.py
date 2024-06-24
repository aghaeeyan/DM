import pandas as pd
from pathlib import Path


dataFile1 = 'rawData/OxCGRT_CAN_differentiated_withnotes_2021.csv' #Source Our world in data https://github.com/OxCGRT/covid-policy-tracker/tree/master/data/Canada
mandate = pd.read_csv(dataFile1, parse_dates=['Date'])
mandate['Date'] = pd.to_datetime(mandate['Date'])
FinalData = pd.DataFrame(columns=['Location','C1','C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'H6', 'H8', 'V4', 'set_date',
                                  'announcement'])
mandate.sort_values(by='Date', inplace=True)
RegionCode = list(mandate['RegionCode'].unique())
RegionCode = [x for x in RegionCode if str(x) != 'nan']
print(RegionCode)
for i in RegionCode:
    date_C1NV = mandate.loc[(mandate['RegionCode'] == i) & (mandate['C1NV_School closing']>1),'Date'].min()
    date_C2NV = mandate.loc[((mandate['RegionCode'] == i) & (mandate['C2NV_Workplace closing']>1)), 'Date'].min()
    date_C3NV = mandate.loc[((mandate['RegionCode'] == i) & (mandate['C3NV_Cancel public events']>1)), 'Date'].min()
    date_C4NV = mandate.loc[((mandate['RegionCode'] == i) & (mandate['C4NV_Restrictions on gatherings'] >1)), 'Date'].min()
    date_C5NV = mandate.loc[((mandate['RegionCode'] == i) & (mandate['C5NV_Close public transport'] >1)), 'Date'].min()
    date_C6NV = mandate.loc[((mandate['RegionCode'] == i) & (mandate['C6NV_Stay at home requirements']>1)), 'Date'].min()
    date_C7NV = mandate.loc[((mandate['RegionCode'] == i) & (mandate['C7NV_Restrictions on internal movement']>1)), 'Date'].min()
    date_H6NV = mandate.loc[((mandate['RegionCode'] == i) & (mandate['H6NV_Facial Coverings']>1)), 'Date'].min()
    date_H8NV = mandate.loc[((mandate['RegionCode'] == i) & (mandate['H8NV_Protection of elderly people']>1)), 'Date'].min()
    date_V4NV = mandate.loc[((mandate['RegionCode'] == i) & (mandate['V4_Mandatory Vaccination (summary)'] == 1)), 'Date'].min()
    vector = [pd.to_datetime("09/09/2021"), (date_C1NV), (date_C2NV), (date_C3NV), (date_C4NV), (date_C5NV),
            (date_C6NV), (date_C7NV), date_H6NV, (date_H8NV), (date_V4NV)]
    announcement = min(vector) - pd.to_timedelta(20, unit='d')
    dummyFile = pd.DataFrame([[i, str(date_C1NV), str(date_C2NV), str(date_C3NV), str(date_C4NV), str(date_C5NV),
                              str(date_C6NV), str(date_C7NV),
                              str(date_H6NV), str(date_H6NV),  str(date_V4NV), min(vector), announcement]],
        columns=['Location', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'H6', 'H8', 'V4', 'set_date', 'announcement'])
    FinalData = pd.concat([dummyFile, FinalData])


filepath1 = Path('Data/mandate_date.csv')
filepath1.parent.mkdir(parents=True, exist_ok=True)
FinalData.to_csv(filepath1)



