from tdm_import import *

# 오프라인 테스트 실행

from tdm_import import *

# self = snubh_cpt_tdm()
# self.offline_execution_main()
# self.generate_tdm_reply_text()
# self.save_result_offline()
# self.open_result_txt()
# calc_text = self.offline_get_interpretation_and_recommendation_text(drug=self.pt_dict['drug'])
# ir_text = self.offline_ir_text_generator(mode='manual', drug=self.pt_dict['drug'])

# """
## 오프라인 테스트 실행
#
# from tdm_import import *
#
# self = snubh_cpt_tdm()
# self.offline_execution_main()
# self.generate_tdm_reply_text()
# self.save_result_offline()
# self.open_result_txt()
# calc_text = self.offline_get_interpretation_and_recommendation_text(drug=self.pt_dict['drug'])
# ir_text = self.offline_ir_text_generator(mode='manual', drug=self.pt_dict['drug'])
# """

def reset_button():
    st.session_state.tdm_date = datetime.today()
    st.session_state.hospital = '병원을 선택하세요'
    st.session_state.tdm_division = '학과를 선택하세요'
    st.session_state.tdm_writer = ''
    st.session_state.ps_viewer = pd.DataFrame(columns=['date', 'drug', 'value'])


def retry_button():
    st.session_state.tdm_inst.retry_execution()

st.set_page_config(page_title="AutoTDM",layout="wide")

with st.sidebar:
    st.title('TDM Practice')

    st.date_input("시행일자", datetime.today(), key='tdm_practice_date')
    st.selectbox('병원', ('병원을 선택하세요','분당서울대학교병원', '서울대학교병원', ), key='hospital')
    st.selectbox('학과', ('학과를 선택하세요','임상약리학과', '진단검사의학과', '약제부',), key='tdm_division')
    st.text_input('사용자명', key='tdm_writer')
    st.session_state['ps_viewer'] = pd.DataFrame(columns=['date', 'drug', 'value'])
    # st.text_input('저장경로', value='C:', key='download_path')

    st.divider()
    scol1, scol2 = st.columns(2, gap='small')
    ## Reset 버튼
    with scol1:
        st.button('Reset', on_click=reset_button)

    ## Retry 버튼
    with scol2:
        st.button('Re-try', on_click=retry_button)

    monitoring_str = ''
    # monitoring_str = '{'
    # for k, v in st.session_state.items():
    #     if k!='monitor': continue
    #     monitoring_str += f"'{k}': '{v}',\n"
    # monitoring_str += '}'
    st.text_area('Memo', monitoring_str, key='memo')

if (st.session_state['hospital']=='병원을 선택하세요') and (st.session_state['tdm_division']=='병원을 선택하세요'):
   pass
else:
   if (st.session_state['hospital'] == '분당서울대학교병원') and (st.session_state['tdm_division'] == '임상약리학과'):
       st.session_state['tdm_inst'] = snubh_cpt_tdm()
       st.session_state['tdm_inst'].execution_flow()

   elif (st.session_state['hospital'] == '서울대학교병원') and (st.session_state['tdm_division'] == '임상약리학과'):
       st.session_state['tdm_inst'] = snuh_cpt_tdm()
       st.session_state['tdm_inst'].exection_flow()
       st.write('TDM 로직을 개발중입니다.')
   else:
       pass





