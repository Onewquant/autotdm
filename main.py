import os, sys

try: project_dir = os.path.abspath(os.path.dirname(__file__))
except: project_dir = os.path.abspath(os.path.dirname("__file__"))
sys.path.append(project_dir)
print(project_dir)

from tdm_import import *

def reset_button():
    st.session_state.hospital = datetime.today()
    st.session_state.hospital = '병원을 선택하세요'
    st.session_state.division = '학과를 선택하세요'
    st.session_state.tdm_user = ''

def retry_button():
    pass

with st.sidebar:
    st.title('TDM Practice')

    st.date_input("시행일자", datetime.today(), key='tdm_practice_date')
    st.selectbox('병원', ('병원을 선택하세요','분당서울대학교병원', '서울대학교병원', ), key='hospital')
    st.selectbox('학과', ('학과를 선택하세요','임상약리학과', '진단검사의학과', '약제부',), key='division')
    st.text_input('사용자명', key='tdm_user')

    st.divider()
    scol1, scol2, scol3 = st.columns(3)
    ## Reset 버튼
    with scol1:
        st.button('Reset', on_click=reset_button)

    ## Retry 버튼
    with scol2:
        retry = st.button('Re-try')
        if retry:
            st.session_state['tdm_inst'].retry_execution()

    monitoring_str = '{'
    for k, v in st.session_state.items():
        monitoring_str += f"'{k}': '{v}',\n"
    monitoring_str += '}'
    st.text_area('모니터링', monitoring_str)

if (st.session_state['hospital']=='병원을 선택하세요') and (st.session_state['division']=='병원을 선택하세요'):
   pass
else:
   if (st.session_state['hospital'] == '분당서울대학교병원') and (st.session_state['division'] == '임상약리학과'):
       st.session_state['tdm_inst'] = snubh_cpt_tdm()
       st.session_state['tdm_inst'].execution_flow()



   elif (st.session_state['hospital'] == '서울대학교병원') and (st.session_state['division'] == '임상약리학과'):
       st.session_state['tdm_inst'] = snuh_cpt_tdm()
       st.session_state['tdm_inst'].exection_flow()
       st.write('TDM 로직을 개발중입니다.')
   else:
       pass






# st.button('Reset', on_click=reset_button)





