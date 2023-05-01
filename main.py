import os, sys

try: project_dir = os.path.abspath(os.path.dirname(__file__))
except: project_dir = os.path.abspath(os.path.dirname("__file__"))
sys.path.append(project_dir)

from tdm_import import *


st.title('TDM practice')

tdm_date = st.date_input("시행일자", datetime.today())
hospital = st.selectbox('병원', ('분당서울대학교병원', '서울대학교병원', ))
division = st.selectbox('학과', ('임상약리학과', '진단검사의학과', '약제부',))
tdm_user = st.text_input('사용자명', '홍길동')

if (hospital=='분당서울대학교병원') and (division=='임상약리학과'):
    tdm_inst = snubh_cpt_tdm(tdm_date, hospital, division, tdm_user)
else:
    st.subheader('선택하신 병원/학과의 TDM 로직은 아직 개발중에 있습니다.')

