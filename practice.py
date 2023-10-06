import pandas as pd
# import psycopg2
#
# csdf = pd.read_excel('C:/Users/ilma0/Downloads/CDM_Concept Set.xlsx')
# csdf[['Name','Concept ID','Concept Name','Domain']]
#
# conn = psycopg2.connect(host="DB주소", dbname="DB이름", user="사용자계정", password="비밀번호", port="포트")
# cur = conn.cursor()
#
# cur.execute("SELECT * FROM cdm2021.MEASUREMENT WHERE ")

collection_df = pd.read_excel('C:/Users/ilma0/PycharmProjects/autotdm/resource/collection.xlsx')
done_df = pd.read_excel('C:/Users/ilma0/PycharmProjects/autotdm/resource/done.xlsx')

collection_df[['id']].merge(done_df[['id','name']], how='left',on=['id']).to_csv('C:/Users/ilma0/PycharmProjects/autotdm/resource/collection_nm.csv',encoding='utf-8-sig', index=False)


collection_df['id'].iloc[0]
done_df['id'].iloc[0]



tdf = pd.read_excel('D:/autotdm/two_departments.xlsx')
tdf.columns
ndf = list()
# for inx, row in tdf.iterrows(): break
for inx, row in tdf.iterrows():
    pnum = int(row['name'][-1])
    pname = row['name'][:-1]

    if pnum==2:
        new_dict = dict()
        for c in ('id','name','sex','age','height', 'weight','tdm_date'):
            if c=='name':
                new_dict[c] = pname
            else:
                new_dict[c] = row[c]
        for c in ('Dose per Adm (mg)','Dosing Interval (h)','Vd (L/kg)','CL (ml/min/kg)','total CL (L/hr)', 'Vd steady state(L)', 'trough concentration', 'AUC (mg*h/L)'):
            new_dict[c+'_2point'] = row[c]
        prev_pname = pname
    if pnum==1:
        if pname==prev_pname:
            for c in ('Vd (L/kg)','CL (ml/min/kg)','total CL (L/hr)', 'Vd steady state(L)', 'trough concentration', 'Daily Dose (mg)', 'AUC (mg*h/L)'):
                new_dict[c+'_1point'] = row[c]
            ndf.append(new_dict)
        else:
            print('Patient 이름이 일치하지 않습니다.')
            raise ValueError

ndf = pd.DataFrame(ndf)
ndf['AUC2'] = ndf['AUC (mg*h/L)_2point'].copy()
ndf['AUC1'] = ndf['AUC (mg*h/L)_1point'].copy()
ndf = ndf.drop(labels=['AUC (mg*h/L)_2point','AUC (mg*h/L)_1point'], axis=1)
ndf.to_csv('D:/autotdm/two_departments_수정.csv', header=True, encoding='utf-8-sig', index=False)