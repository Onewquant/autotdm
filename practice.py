import pandas as pd
import psycopg2

csdf = pd.read_excel('C:/Users/ilma0/Downloads/CDM_Concept Set.xlsx')
csdf[['Name','Concept ID','Concept Name','Domain']]

conn = psycopg2.connect(host="DB주소", dbname="DB이름", user="사용자계정", password="비밀번호", port="포트")
cur = conn.cursor()

cur.execute("SELECT * FROM cdm2021.MEASUREMENT WHERE ")