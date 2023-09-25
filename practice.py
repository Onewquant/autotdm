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