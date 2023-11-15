import re, os
import pandas as pd
import numpy as np
import streamlit
import streamlit.proto.Spinner_pb2

from tdm_formats.tdm import *
from datetime import datetime, timedelta
import json


def check_dir(dir_path):
    if os.path.exists(dir_path):
        pass
    else:
        os.mkdir(dir_path)

def check_dir_continuous(dir_list, root_path='.'):
    dir_str = root_path
    for direc in dir_list:
        dir_str += f'/{direc}'
        check_dir(dir_str)
    return dir_str


def open_result_txt(drug='', name='', id='', tdm_date='', auto=True):
    os.popen(f'call {os.getcwd()}\\result\\reply_text\\{drug}_{name}_{id}_{tdm_date}.txt')

def get_result_text_from_file(drug='', name='', id='', tdm_date='', rootpath=f'{os.getcwd()}\\result\\reply_text'):
    filename = f'{drug}_{name}_{id}_{tdm_date}.txt'
    filepath = rootpath + '\\' + filename
    with open(filepath, "r", encoding="utf-8-sig") as f:
        rtext = f.read()

def get_tdm_dateform(date_str):
    dstr_list = list()
    for dstr in date_str.split('-')[1:]:
        if dstr not in ('10', '20', '30'):
            dstr_list.append(dstr.replace('0', ''))
        else:
            dstr_list.append(dstr)

    return f"{dstr_list[0]}/{dstr_list[1]}"

def determine_int_or_float(value):
    return int(value) if value * 1000 == int(value) * 1000 else value

def parse_daily_notnull_labdf_into_addtext(lab_date, lab_key, not_null_df):
    addtext = ''
    ud_split = lab_date.split('-')
    ud_text = f"({ud_split[1].replace('0', '') if ud_split[1] not in ('10','20','30') else ud_split[1]}/{ud_split[2].replace('0', '') if ud_split[2] not in ('10','20','30') else ud_split[2]})"

    if lab_key == 'WBC(seg%)/ANC':
        d_wbc, d_segneut, d_anc = np.nan, np.nan, np.nan
        # for nninx, nnrow in non_null_frag.iterrows(): break
        # not_null_df
        all_valid_df = not_null_df[~not_null_df.isnull().any(axis=1)]
        if len(all_valid_df)>0:
            nnrow = all_valid_df.iloc[0]
            d_wbc = round(nnrow['WBC'] * 1000,5)
            d_segneut = nnrow['Seg.neut.']
            d_anc = nnrow['ANC']
        else:
            for nninx, nnrow in not_null_df.iterrows():
                if np.isnan(d_wbc): d_wbc = round(nnrow['WBC'] * 1000)
                if np.isnan(d_segneut): d_segneut = nnrow['Seg.neut.']
                if np.isnan(d_anc): d_anc = nnrow['ANC']

        if np.isnan(d_wbc): d_wbc = '.'
        else: d_wbc = determine_int_or_float(value=d_wbc)
        if np.isnan(d_segneut): d_segneut = '.'
        else: d_segneut = determine_int_or_float(value=d_segneut)
        if np.isnan(d_anc): d_anc = '.'
        else: d_anc = determine_int_or_float(value=d_anc)

        addtext = f"{ud_text}{d_wbc}({d_segneut})/{d_anc} <-"
    elif lab_key == 'Ca/K':
        d_ca, d_k = np.nan, np.nan
        # for nninx, nnrow in non_null_frag.iterrows(): break
        all_valid_df = not_null_df[~not_null_df.isnull().any(axis=1)]
        if len(all_valid_df) > 0:
            nnrow = all_valid_df.iloc[0]
            d_ca = nnrow['Ca, total']
            d_k = nnrow['K']
        else:
            for nninx, nnrow in not_null_df.iterrows():
                if np.isnan(d_ca): d_ca = nnrow['Ca, total']
                if np.isnan(d_k): d_k = nnrow['K']

        if np.isnan(d_ca): d_ca = '.'
        else: d_ca = determine_int_or_float(value=d_ca)
        if np.isnan(d_k): d_k = '.'
        else: d_k = determine_int_or_float(value=d_k)

        addtext = f"{ud_text}{d_ca}/{d_k} <-"
    elif lab_key == 'PLT/PT/aPTT':
        plt, pt_pct, aptt = np.nan, np.nan, np.nan
        # for nninx, nnrow in non_null_frag.iterrows(): break
        all_valid_df = not_null_df[~not_null_df.isnull().any(axis=1)]
        if len(all_valid_df) > 0:
            nnrow = all_valid_df.iloc[0]
            plt = nnrow['Platelet']
            pt_pct = nnrow['PT %']
            aptt = nnrow['aPTT']
        else:
            for nninx, nnrow in not_null_df.iterrows():
                if np.isnan(plt): plt = nnrow['Platelet']
                if np.isnan(pt_pct): pt_pct = nnrow['PT %']
                if np.isnan(aptt): aptt = nnrow['aPTT']

        if np.isnan(plt): plt = '.'
        else: plt = determine_int_or_float(value=plt)
        if np.isnan(pt_pct): pt_pct = '.'
        else: pt_pct = determine_int_or_float(value=pt_pct)
        if np.isnan(aptt): aptt = '.'
        else: aptt = determine_int_or_float(value=aptt)

        addtext = f"{ud_text}{plt}k/{pt_pct}/{aptt} <-".replace('.k/','./')

    elif lab_key == 'T.bil/AST/ALT':
        tbil, ast, alt = np.nan, np.nan, np.nan
        # for nninx, nnrow in non_null_frag.iterrows(): break
        all_valid_df = not_null_df[~not_null_df.isnull().any(axis=1)]
        if len(all_valid_df) > 0:
            nnrow = all_valid_df.iloc[0]
            tbil = nnrow['T.B']
            ast = nnrow['AST']
            alt = nnrow['ALT']
        else:
            for nninx, nnrow in not_null_df.iterrows():
                if np.isnan(tbil): tbil = nnrow['T.B']
                if np.isnan(ast): ast = nnrow['AST']
                if np.isnan(alt): alt = nnrow['ALT']

        if np.isnan(tbil): tbil = '.'
        else: tbil = determine_int_or_float(value=tbil)
        if np.isnan(ast): ast = '.'
        else: ast = determine_int_or_float(value=ast)
        if np.isnan(alt): alt = '.'
        else: alt = determine_int_or_float(value=alt)

        addtext = f"{ud_text}{tbil}/{ast}/{alt} <-"

    elif lab_key == 'BUN/Cr':
        d_bun, d_cr = np.nan, np.nan
        # for nninx, nnrow in non_null_frag.iterrows(): break
        all_valid_df = not_null_df[~not_null_df.isnull().any(axis=1)]
        if len(all_valid_df) > 0:
            nnrow = all_valid_df.iloc[0]
            d_bun = nnrow['BUN']
            d_cr = nnrow['Cr (S)']
        else:
            for nninx, nnrow in not_null_df.iterrows():
                if np.isnan(d_bun): d_bun = nnrow['BUN']
                if np.isnan(d_cr): d_cr = nnrow['Cr (S)']

        if np.isnan(d_bun): d_bun = '.'
        else: d_bun = determine_int_or_float(value=d_bun)
        if np.isnan(d_cr): d_cr = '.'
        else: d_cr = determine_int_or_float(value=d_cr)

        addtext = f"{ud_text}{d_bun}/{d_cr} <-"

    elif lab_key == 'GFR':
        d_gfr_mdrd, d_gfr_ckdepi, d_gfr_schwartz = np.nan, np.nan, np.nan
        d_age = not_null_df['나이'].iloc[0]
        # for nninx, nnrow in non_null_frag.iterrows(): break
        exist_colnames = list(not_null_df.columns)
        mdrd_existence = ('eGFR-MDRD' in exist_colnames)
        ckdepi_existence = ('eGFR-CKD-EPI' in exist_colnames)
        schwartz_existence = ('eGFR-Schwartz(소아)' in exist_colnames)

        for nninx, nnrow in not_null_df.iterrows():
            if np.isnan(d_gfr_mdrd) and mdrd_existence : d_gfr_mdrd = nnrow['eGFR-MDRD']
            if np.isnan(d_gfr_ckdepi) and ckdepi_existence : d_gfr_ckdepi = nnrow['eGFR-CKD-EPI']
            if np.isnan(d_gfr_schwartz) and schwartz_existence : d_gfr_schwartz = nnrow['eGFR-Schwartz(소아)']

        if np.isnan(d_gfr_mdrd): d_gfr_mdrd = '.'
        if np.isnan(d_gfr_ckdepi): d_gfr_ckdepi = '.'
        if np.isnan(d_gfr_schwartz): d_gfr_schwartz = '.'
        if (d_age <= 18):
            if schwartz_existence:
                d_gfr = d_gfr_schwartz
            else:
                d_gfr = '.'
        else:

            if mdrd_existence and ckdepi_existence: d_gfr = d_gfr_mdrd
            elif mdrd_existence and (not ckdepi_existence): d_gfr = d_gfr_mdrd
            elif (not mdrd_existence) and ckdepi_existence: d_gfr = d_gfr_ckdepi
            else: d_gfr ='.'

        if type(d_gfr)==float:
            d_gfr = determine_int_or_float(value=d_gfr)

        addtext = f"{ud_text}{d_gfr} <-"
    elif lab_key == 'CRP':
        d_crp = np.nan
        # for nninx, nnrow in non_null_frag.iterrows(): break
        for nninx, nnrow in not_null_df.iterrows():
            if np.isnan(d_crp): d_crp = nnrow['CRP']

        if np.isnan(d_crp): d_crp = '.'
        else: d_crp = determine_int_or_float(value=d_crp)

        addtext = f"{ud_text}{d_crp} <-"
    elif lab_key == 'Alb':
        d_alb = np.nan
        # for nninx, nnrow in non_null_frag.iterrows(): break
        for nninx, nnrow in not_null_df.iterrows():
            if np.isnan(d_alb): d_alb = nnrow['Albumin']

        if np.isnan(d_alb): d_alb = '.'
        else: d_alb = determine_int_or_float(value=d_alb)

        addtext = f"{ud_text}{d_alb} <-"

    elif lab_key == 'Ammonia':
        d_ammo = np.nan
        # for nninx, nnrow in non_null_frag.iterrows(): break
        for nninx, nnrow in not_null_df.iterrows():
            if np.isnan(d_ammo): d_ammo = nnrow['Ammo']

        if np.isnan(d_ammo): d_ammo = '.'
        else: d_ammo = determine_int_or_float(value=d_ammo)

        addtext = f"{ud_text}{d_ammo} <-"
    return addtext

def get_all_conomitant_drugs(order_str):
    raw_order_cols = ['비고','처방지시', '발행처', '발행의', '수납', '약국/검사', '주사시행처', 'Acting', '변경의']
    result_order_cols = ['date', 'time', '처방지시', '발행처', '발행의', '수납', '약국/검사', '주사시행처', 'Acting', '변경의', 'D/C',
                              '보류', '반납']
    order_df = pd.DataFrame(columns=result_order_cols)

    raw_order_str_list = order_str.split('\n')

    parsed_order_list = list()
    for row in raw_order_str_list:
        parsed_order_list.append(dict(list(zip(raw_order_cols, row.split('\t')))))

    order_df = pd.DataFrame(parsed_order_list)

    order_df['date'] = ''
    order_df['time'] = ''
    order_df['D/C'] = ''
    order_df['보류'] = ''
    order_df['반납'] = ''
    for inx, row in order_df.iterrows():
        order_df.at[inx, '처방지시'] = row['처방지시'].strip()
        order_df.at[inx, 'date'] = row['발행의'].split(' ')[-2]
        order_df.at[inx, 'time'] = row['발행의'].split(' ')[-1]
        for order_state in ('D/C', '보류', '반납'):
            order_df.at[inx, order_state] = order_state in row['처방지시']

    order_df = order_df[result_order_cols]

    dodf = order_df.copy()
    dodf2 = dodf[dodf['Acting'].map(lambda x:('Y' in x) or ('Z' in x) or ('C' in x))].copy()
    dodf2['처방지시'] = dodf2['처방지시'].map(lambda x:x.replace('[반납]','').replace('[Dr.확인후]','').replace('[보류]','').replace('[Prn]','').replace('[D/C]','').replace('[Em]','').replace('  ',' '))
    dodf2 = dodf2[dodf2['처방지시'].map(lambda x:('치료유동경관미음' not in x) and ('BST' not in x) and ('Body weight check' not in x) and ('head elevation' not in x) and ('DVT prophylaxis IPC' not in x) and ('Circumference check' not in x) and ('Restraint apply' not in x) and ('4-extremities BP check' not in x) and ('Hemodialysis Diet' not in x) and ('타과의뢰  : TO' not in x))]
    dodf2['처방지시'] = np.where(dodf2['처방지시'].map(lambda x:x.split(' ')[0][-1]==')'),dodf2['처방지시'].map(lambda x:''.join(x.split(')')[1:]).strip()), dodf2['처방지시'])

    dodf2['처방SC'] = dodf2['처방지시'].map(lambda x:x.split(' ')[0])
    tsc_dict = dict()
    for inx, row in dodf2.drop_duplicates(subset=['처방SC']).iterrows():
        tsc_dict[row['처방SC']] = row['처방지시']
    dodf2 = dodf2.reset_index(drop=True)
    result_df = list()
    for k,v in tsc_dict.items():
        tddf = dodf2[dodf2['처방SC']==k].copy()
        result_df.append({'처방ShortCode':' '.join(v.split(' ')[:3]), '처방Full':v, '사용날짜List':list(tddf['date'].unique())})
    result_df = pd.DataFrame(result_df)

    result_df['성분명'] = result_df['처방Full'].map(lambda x: x.split('(')[1].split('[')[0] if len(x.split('(')) > 1 else x.split('[')[0])
    result_df = result_df[['성분명', '처방ShortCode', '처방Full', '사용날짜List']]

    result_df = result_df.sort_values(['성분명'], ignore_index=True)

    return result_df

@st.cache_data(ttl=60)
def convert_df(df):
    return df.to_csv(index=False, encoding='utf-8-sig')

class snubh_cpt_tdm(tdm):
    def __init__(self):
        super().__init__()
        self.individual_vars()
        # self.set_basic_info(tdm_date, hospital, division, tdm_user)

    def download_button_manager(self, mode=''):

        # download_root_dir = f"{st.session_state['download_path']}"
        download_root_dir = f"D:"
        try: check_dir(dir_path=download_root_dir)
        except: download_root_dir = f"C:"

        if mode=='result':
            # check_dir_continuous(['autotdm','result'], root_path=download_root_dir)
            # filename = f"{self.short_drugname_dict[st.session_state['drug']]}_{st.session_state['name']}_{st.session_state['id']}_{datetime.strftime(st.session_state['tdm_date'], '%Y%m%d')}.txt"
            # download_path = f"{download_root_dir}/autotdm/result/{filename}"
            #
            # with open(download_path, "w", encoding="utf-8-sig") as f:
            #     f.write(st.session_state['first_draft'])

            try:
                check_dir_continuous(['autotdm',], root_path=download_root_dir)
                filename = f"two_point_research.csv"
                file_path = f"{download_root_dir}/autotdm/{filename}"

                round_num = 3
                vd_val = float(round(st.session_state['vd_ss'] / st.session_state['weight'], round_num))
                cl_val = float(round(st.session_state['total_cl'] * 1000 / 60 / st.session_state['weight'], round_num))
                total_cl_val = st.session_state['total_cl']
                vdss_val = st.session_state['vd_ss']
                dailydose_val = round(st.session_state['adm_amount'] * (24 / st.session_state['adm_interval']), round_num)
                auc_val = round((st.session_state['adm_amount'] * (24 / st.session_state['adm_interval'])) / round(st.session_state['total_cl'], round_num), round_num)
                trough_conc = st.session_state['est_trough']

                result_dict = dict([(sscol, st.session_state[sscol]) for sscol in ('id', 'name', 'sex', 'age', 'height', 'weight', 'tdm_date')])
                result_dict['Dose per Adm (mg)'] = st.session_state['adm_amount']
                result_dict['Dosing Interval (h)'] = st.session_state['adm_interval']
                result_dict['Vd (L/kg)'] = vd_val
                result_dict['CL (ml/min/kg)'] = cl_val
                result_dict['total CL (L/hr)'] = total_cl_val
                result_dict['Vd steady state(L)'] = vdss_val
                result_dict['Daily Dose (mg)'] = dailydose_val
                result_dict['AUC (mg*h/L)'] = auc_val
                result_dict['trough concentration'] = trough_conc
                result_df = pd.DataFrame([result_dict])

                result_df.to_csv(file_path, encoding='utf-8-sig', mode='a', index=False, header=(not os.path.exists(file_path)))

                st.success(f"{st.session_state['id']} / {st.session_state['name']} / Result / Rec Successfully", icon=None)
            except:
                st.error(f"{st.session_state['id']} / {st.session_state['name']} / Result / Rec Failed", icon=None)

        elif mode=='input_records':
            input_record_dirname =f"{self.short_drugname_dict[st.session_state['drug']]}_{st.session_state['name']}({st.session_state['id']}){st.session_state['sex']}{st.session_state['age']}({datetime.strftime(st.session_state['tdm_date'], '%Y%m%d')})"
            check_dir_continuous(['autotdm', 'input_records', input_record_dirname], root_path=download_root_dir)
            for key in ('history', 'lab', 'vs', 'order'):
                filename = f"{key}_{st.session_state['name']}.txt"
                download_path = f"{download_root_dir}/autotdm/input_records/{input_record_dirname}/{filename}"
                st.session_state['memo'] = download_path
                with open(download_path, "w", encoding="utf-8-sig") as f:
                    f.write(st.session_state[key])



    def execution_flow(self):

        self.rcol1, self.rcol2 = st.columns([1,2], gap="medium")

        with self.rcol1:
            for k, v in self.basic_pt_term_dict.items():
                if k=='tdm_date':
                    st.date_input(v, datetime.today(), key=k)
                elif k=='sex':
                    st.radio(v,options=('남','여'), horizontal=True, key=k)
                    continue
                elif k=='age':
                    st.number_input(label=v, min_value=0 ,max_value=120, step=1, key=k)
                    if st.session_state['age'] <= 18:
                        st.session_state['pedi'] = True
                        self.pt_dict['pedi'] = True
                    else:
                        st.session_state['pedi'] = False
                        self.pt_dict['pedi'] = False
                elif k in ('height','weight'):
                    st.number_input(label=v, min_value=0.1 ,max_value=300.0, step=1.0, key=k)
                elif k=='pedi':
                    continue
                elif k=='drug':
                    st.selectbox(v, ('약물을 입력하세요','Vancomycin', 'Amikacin', 'Gentamicin', 'Digoxin', 'Valproic Acid', 'Phenytoin'), key=k)
                    continue
                else:
                    st.text_input(v, key=k)

            st.divider()

        if st.session_state['drug']!='약물을 입력하세요':

            self.tdm_writer = st.session_state['tdm_writer']
            self.tdm_date = st.session_state['tdm_date'].strftime('%Y-%m-%d')
            self.pt_dict['tdm_date'] = self.tdm_date
            self.pt_dict['drug'] = self.short_drugname_dict[st.session_state['drug']]

            with self.rcol1:

                st.write(f"<Ref> {st.session_state['id']} / {st.session_state['name']} / {st.session_state['drug']} TDM")

                additional_inputs = self.additional_pt_term_dict[self.short_drugname_dict[st.session_state['drug']]]
                # if len(additional_inputs)==0: pass
                # else:
                for k, v in additional_inputs.items():
                    if k == 'consult':
                        # st.session_state[k]= self.parse_patient_history(hx_df=self.pt_hx_df, cont_type=k)
                        continue
                    else:
                        st.text_area(v, key=k)

                st.button('Generate the first draft', on_click=self.execution_of_generating_first_draft, key='generate_first_draft')

            with self.rcol2:

                st.text_area(label='Draft', value='', height=594, key='first_draft')

                self.rcol3, self.rcol4, self.rcol5 = st.columns([1, 2, 3], gap="medium")

                with self.rcol3:
                    st.download_button(label='Download', data=st.session_state['first_draft'], file_name=f"{self.short_drugname_dict[st.session_state['drug']]}_{st.session_state['name']}_{st.session_state['id']}_{datetime.strftime(st.session_state['tdm_date'],'%Y%m%d')}.txt")
                with self.rcol4:
                    st.button('Rec for Research', on_click=self.download_button_manager, args=('result',), key='rec_for_research')
                with self.rcol5:
                    # st.session_state['drug_use_df'] = pd.DataFrame(columns=['date', 'drug', 'value'])
                    st.download_button(label='DDI Evaluation', data=convert_df(st.session_state['ps_viewer']), file_name=f"(DDI_EVAL) {self.short_drugname_dict[st.session_state['drug']]}_{st.session_state['name']}_{st.session_state['id']}_{datetime.strftime(st.session_state['tdm_date'],'%Y%m%d')}.csv", mime="text/csv", on_click=self.patient_state_viewer_analysis, args=('/Y',) , key='ddi_eval_download')

                st.divider()

                st.write(f"<PK parameters 입력>")

                self.define_ir_info()

                for k, v in self.ir_term_dict[self.short_drugname_dict[st.session_state['drug']]].items():
                    if (self.short_drugname_dict[st.session_state['drug']]=='VCM') and (k=='vc') and (st.session_state['age'] <= 18):
                        st.session_state[k] = '-'
                        continue
                    else: st.number_input(label=v, min_value=0.1, max_value=1500.0, step=1.0, key=k)

                st.button('Reflect Parameters', on_click=self.reflecting_parameters, key='reflect_parameters')

                st.divider()

                st.write(f"<Concentration Level & Recommendations>")

                self.ir_drug_dict = self.ir_recomm_dict[self.short_drugname_dict[st.session_state['drug']]]

                st.selectbox('농도Level', ['선택하세요',]+list(self.ir_drug_dict.keys()), key='ir_conc')

                if st.session_state['ir_conc']!='선택하세요':

                    self.recom_tot_dict = self.ir_drug_dict[st.session_state['ir_conc']]

                    st.selectbox('항정상태여부', ['선택하세요', ] + [k.split('_')[1] for k in self.recom_tot_dict.keys() if k.split('_')[0]=='rec1'], key='ir_state')

                    st.selectbox('Method', ['선택하세요', ] + [k.split('_')[1] for k in self.recom_tot_dict.keys() if k.split('_')[0]=='rec2'], key='ir_method')

                    st.button('Reflect IR', on_click=self.reflecting_ir_text, key='reflect_ir')

    def execution_of_generating_first_draft(self):
        # self.tdm_writer = st.session_state['tdm_writer']
        # self.tdm_date = st.session_state['tdm_date'].strftime('%Y-%m-%d')
        # self.pt_dict['tdm_date'] = self.tdm_date
        # self.pt_dict['drug'] = self.short_drugname_dict[st.session_state['drug']]

        # self.download_button_manager(mode="input_records")
        # try:

            for k, v in st.session_state.items():
                if k in ('tdm_inst', 'tdm_date', 'drug', 'first_draft'):continue
                elif k=='sex':
                    self.pt_dict[k]= 'M' if v=='남' else 'F'
                elif k=='age':
                    self.pt_dict[k] = v
                elif k=='history':
                    self.pt_hx_raw = self.get_reduced_sentence(v)
                    if self.pt_hx_raw != '':
                        self.pt_hx_df = self.get_pt_hx_df(hx_str=self.pt_hx_raw)
                        # st.session_state['monitor'] = self.pt_hx_df
                        # st.session_state['monitor'] = self.pt_dict['pedi']
                        self.pt_dict[k] = self.parse_patient_history(hx_df=self.pt_hx_df, cont_type=k)
                        # st.session_state['monitor'] = self.parse_patient_history(hx_df=self.pt_hx_df, cont_type='consult')
                        self.pt_dict['consult'] = self.parse_patient_history(hx_df=self.pt_hx_df, cont_type='consult')
                    else:
                        self.pt_dict[k] = ''
                        self.pt_dict['consult'] = ''
                elif k == 'hemodialysis':
                    self.pt_dict[k] = self.get_reduced_sentence(v)
                elif k == 'electroencephalography':
                    self.pt_dict[k] = self.get_parsed_eeg_result(eeg_result=v)
                elif k == 'echocardiography':
                    self.pt_dict[k] = self.get_parsed_echocardiography_result(echo_result=v)
                elif k=='ecg':
                    self.pt_dict[k] = self.get_parsed_ecg_result(ecg_result=v)
                elif k=='vs':
                    self.pt_dict[k] = self.parse_vs_record(raw_vs=v)
                elif k=='lab':
                    self.pt_dict[k] = self.get_parsed_lab_df(value=v)
                elif k=='order':
                    self.pt_dict[k] = self.parse_order_record(order_str=v)
                else:
                    self.pt_dict[k] = v

            self.generate_tdm_reply_text()
            st.session_state['first_draft'] = self.file_content

            # self.order_data_prep_for_ddi_analysis()

            # crst_str = 'drugs\t'
            # for c in list(st.session_state['drug_use_df'].columns):
            #     crst_str+= f'{c}\t'
            # crst_str+='\n'
            # for inx, row in st.session_state['drug_use_df'].iterrows():
            #     crst_str += f'{inx}\t'
            #     for rc_inx, rc_val in enumerate(row):
            #         crst_str += f'{rc_val}\t'
            #     crst_str += '\n'
            #
            # st.session_state['first_draft'] = crst_str
            # st.session_state['first_draft'] = str(st.session_state['drug_use_df'])


        # except:
        #     st.error(f"{st.session_state['id']} / {st.session_state['name']} / 1st Draft / Generation Failed", icon=None)
        # st.text_area('',self.file_content,)

    def retry_execution(self):

        if st.session_state.tdm_date == datetime.today(): pass
        else: st.session_state.tdm_date = datetime.today()
        st.session_state['id'] = ''
        st.session_state['name'] = ''
        st.session_state['sex'] = '남'
        st.session_state['age'] = 1
        st.session_state['height'] = 1.0
        st.session_state['weight'] = 1.0
        st.session_state['drug'] = '약물을 입력하세요'


    def individual_vars(self):
        self.prev_date = ''
        self.win_period = 7
        self.basic_pt_term_dict = {'tdm_date': 'TDM날짜',
                                   'id': '환자등록번호',
                                   'name': '이름',
                                   'sex': '성별',
                                   'age': '나이',
                                   'pedi': '소아여부',
                                   'height': '키',
                                   'weight': '체중',
                                   'drug': '약물',
                                   }
        self.basic_pt_dict = dict()
        for k, v in self.basic_pt_term_dict.items(): self.basic_pt_dict[k] = ''

        self.additional_pt_term_dict = {'DRUG was not chosen':dict(),
                                        'VCM': {'history': '환자 History',
                                                'hemodialysis': '혈액투석',
                                                'consult': '타과컨설트',
                                                'vs': 'Vital Sign - (BT))',
                                                'lab': 'Lab 검사결과',
                                                'order': '전체오더'
                                                },
                                        'DGX': {'history': '환자 History',
                                                'echocardiography': '심장초음파',
                                                'ecg': '심전도판독결과',
                                                'consult': '타과컨설트',
                                                'vs': 'Vital Sign - (SBP/DBP/HR)',
                                                'lab': 'Lab 검사결과',
                                                'order': '전체오더'
                                                },
                                        'AMK': {'history': '환자 History',
                                                'hemodialysis': '혈액투석',
                                                'consult': '타과컨설트',
                                                'vs': 'Vital Sign - (BT)',
                                                'lab': 'Lab 검사결과',
                                                'order': '전체오더'
                                                },
                                        'GTM': {'history': '환자 History',
                                                'hemodialysis': '혈액투석',
                                                'consult': '타과컨설트',
                                                'vs': 'Vital Sign - (BT)',
                                                'lab': 'Lab 검사결과',
                                                'order': '전체오더'
                                                },
                                        'VPA': {'history': '환자 History',
                                                'electroencephalography': '뇌전도검사',
                                                'consult': '타과컨설트',
                                                'lab': 'Lab 검사결과',
                                                'order': '전체오더'
                                                },
                                        }

        self.short_drugname_dict = {'약물을 입력하세요':'DRUG was not chosen', 'Vancomycin': 'VCM', 'Amikacin': 'AMK', 'Gentamicin': 'GTM', 'Digoxin': 'DGX', 'Valproic Acid': 'VPA', 'Phenytoin': 'PHT'}
        # self.short_drugname = ''

        self.pt_term_dict = dict()
        self.pt_dict = dict()

        self.ir_term_dict = dict()
        self.ir_dict = dict()

        self.hx_type_tups = (
        '타과의뢰', '타과회신', '입원초진', '입원경과', '수술전기록', '수술기록', '수술실퇴실전기록', '마취전평가', '마취기록', '과별서식', '응급초진', '응급경과',
        '응급실 퇴실기록')
        self.pt_hx_raw = ''
        self.pt_hx_df = pd.DataFrame(columns=['type', 'department', 'date', 'text'])

        self.raw_order_cols = ['비고','처방지시', '발행처', '발행의', '수납', '약국/검사', '주사시행처', 'Acting', '변경의']
        self.result_order_cols = ['date', 'time', '처방지시', '발행처', '발행의', '수납', '약국/검사', '주사시행처', 'Acting', '변경의', 'D/C',
                                  '보류', '반납']
        self.order_df = pd.DataFrame(columns=self.result_order_cols)
        # self.order_df = pd.DataFrame(columns=['date', 'dt', 'order', 'acting', 'completion'])

        self.raw_vs_cols = ['SBP (mmHg)', 'DBP (mmHg)', 'PR (회/min)']
        self.vs_df = pd.DataFrame(columns=self.raw_vs_cols)

        self.drug_list = ('VCM', 'VPA', 'TBM', 'PHT', 'THP', 'PBT', 'DGX', 'CBZ', 'AMK', 'GTM')
        self.drug_consult_dict = {
            'VCM': '감염내과',
            'VPA': '신경과',
            'TBM': '감염내과',
            'PHT': '신경과',
            'THP': '',
            'PBT': '신경과',
            'DGX': '순환기내과',
            'CBZ': '신경과',
            'AMK': '감염내과',
            'GTM': '감염내과',
        }

        self.drug_fullname_dict = {'VCM': ['VANCOMYCIN', ],
                                   'DGX': ['DIGOXIN', 'DIGOSIN'],
                                   'AMK': ['AMIKACIN', ],
                                   'VPA': ['ORFIL', 'DEPAKOTE', 'DEPAKIN'],
                                   'GTM': ['GENTAMICIN', ],
                                   }

        self.tdm_target_txt_dict = {'VCM': '400-600 mg*h/L',
                                    'DGX': '0.5-1.5 ng/mL',
                                    'AMK': 'Peak > 25, Trough < 5 ㎍/mL',
                                    'VPA': '50-100 ㎍/mL',
                                    'GTM': 'Peak > 5, Trough < 1 ㎍/mL', }

        # self.result_dir = f"{project_dir}/result"
        # self.resource_dir = f"{project_dir}/resource"
        # self.inputfiles_dir = f"{project_dir}/input_files"
        # self.inputrecord_dir = f"{project_dir}/input_records"
        # self.lab_inputfile_path = f"{self.inputfiles_dir}/lab_input.xlsx"
        # self.reply_text_saving_dir = f"{self.result_dir}/reply_text"
        #
        # for cdir in self.result_dir, self.resource_dir, self.inputfiles_dir, self.inputrecord_dir: check_dir(cdir)
        # for cdir in self.lab_inputfile_path, self.reply_text_saving_dir: check_dir(cdir)

        self.ldf = pd.DataFrame(columns=['no_lab'])
        self.raw_lab_input = 'N'

    def get_parsed_lab_df(self, value):
        # value=''
        # value=input()
        # import pandas as pd
        raw_ldf_cols = ['보고일', '오더일', '검사명', '검사결과', '직전결과', '참고치', '결과비고', '오더비고']
        raw_ldf = pd.DataFrame([tbl_row.split('\t') for tbl_row in value.split('\n') if tbl_row!=''])
        cur_rldf_cols = list(raw_ldf.columns)
        vld_rldf_cols = list()
        for i in range(len(cur_rldf_cols)):
            if i + 1 > len(cur_rldf_cols):
                vld_rldf_cols.append(str(i))
            else:
                vld_rldf_cols.append(raw_ldf_cols[i])

        raw_ldf.columns = vld_rldf_cols
        if (len(raw_ldf.columns)==1) or (len(raw_ldf)<=1):
            self.ldf = pd.DataFrame(columns=['date','dt'])
            return self.ldf
        # self.ldf[~self.ldf['Creatinine'].isna()]


        for inx, rrow in raw_ldf.iterrows():
            if (rrow['검사명'] == 'WBC') and ('HPF' in rrow['참고치']):
                raw_ldf.at[inx, '검사명'] = 'u.WBC'
            elif (rrow['검사명'] == 'WBC') and ('mm³' in rrow['참고치']):
                raw_ldf.at[inx, '검사명'] = 'em.WBC'
            elif (rrow['검사명'] == 'RBC') and ('HPF' in rrow['참고치']):
                raw_ldf.at[inx, '검사명'] = 'u.RBC'


        # raw_ldf['검사명'].unique()
        # raw_ldf['date'] =
        raw_ldf['검사명'] = raw_ldf['검사명'].map(lambda x: x.strip())
        raw_ldf['dt_raw'] = raw_ldf[['보고일', '오더일']].min(axis=1) + 'T00:00:00'
        raw_ldf_list = list()
        # for dt, frag_ldf in raw_ldf[['dt_raw', '검사명', '검사결과']].groupby(['dt_raw']):break
        for dt, frag_ldf in raw_ldf[['dt_raw', '검사명', '검사결과']].groupby(['dt_raw']):
            frag_ldf = frag_ldf.reset_index(drop=True)
            frag_ldf['index'] = frag_ldf.index
            frag_ldf['dt'] = frag_ldf[['dt_raw', 'index']].apply(lambda x: (
                        datetime.strptime(x['dt_raw'], '%Y-%m-%dT%H:%M:%S') + timedelta(
                    seconds=int(x['index']))).strftime('%Y-%m-%dT%H:%M:%S'), axis=1)
            #
            # for smdtlab_dict in [{'WBC':0, 'Seg.neut.':0, 'ANC':0}, {'Ca':0, 'total':0, 'K':0}, {'BUN':0, 'Cr (S)':0}]: break
            for smdtlab_dict in [{'WBC': 0, 'Seg.neut.': 0, 'ANC': 0}, {'Ca, total': 0, 'K': 0},
                                 {'BUN': 0, 'Cr (S)': 0, 'Creatinine': 0}, {'Platelet': 0, 'PT %': 0, 'aPTT': 0}]:
                smdtlab_df = frag_ldf[frag_ldf['검사명'].isin(smdtlab_dict.keys())].copy()
                if len(smdtlab_df) == 0: continue
                smdt = smdtlab_df['dt'].min()
                smdtlab_df['dt'] = smdt
                # for smdinx, smdrow in smdtlab_df.iterrows():break
                for smdinx, smdrow in smdtlab_df.iterrows():
                    frag_ldf.at[smdinx, 'dt'] = (datetime.strptime(smdrow['dt'], '%Y-%m-%dT%H:%M:%S') + timedelta(
                        seconds=smdtlab_dict[smdrow['검사명']])).strftime('%Y-%m-%dT%H:%M:%S')
                    smdtlab_dict[smdrow['검사명']] += 1

            raw_ldf_list.append(frag_ldf[['dt', '검사명', '검사결과']].copy())
        raw_ldf = pd.concat(raw_ldf_list, ignore_index=True).reset_index(drop=True)

        raw_ldf = raw_ldf.pivot(index='dt', columns='검사명', values='검사결과')
        raw_ldf.columns.name = None
        raw_ldf = raw_ldf.reset_index(drop=False)
        # self.re_ldf.columns
        self.re_ldf = raw_ldf.copy()

        # self.re_ldf = self.re_ldf.reset_index(drop=False).rename(columns={'index':'dt'})
        # self.re_ldf['dt'] = self.re_ldf['dt'].map(lambda x:x.strftime('%Y-%m-%dT%H:%M:%S'))
        self.re_ldf['date'] = self.re_ldf['dt'].map(lambda x: x.split('T')[0])
        raw_clist = list(self.re_ldf.columns)
        raw_cser = pd.Series(raw_clist)
        raw_cset = set(self.re_ldf.columns)

        dup_cols = [c for c in raw_cset if (raw_cser == c).sum() > 1]
        uniq_cols = list(raw_cset - set(dup_cols))
        res_cols = list(raw_cser.drop_duplicates(keep='first'))

        # for c in dup_cols[1:]: break
        self.ldf = self.re_ldf[uniq_cols].copy()
        for c in dup_cols:
            newc_ds = pd.Series(np.full(len(self.re_ldf), np.nan), index=list(self.re_ldf.index))
            for cinx, crow in self.re_ldf[c].iterrows():
                # crow = self.re_ldf[c].iloc[0]
                crow_uniq = crow.drop_duplicates(keep='first')
                crow_uniq = crow_uniq.map(lambda x: x if type(x) == float else np.nan).drop_duplicates(keep='first')
                nv_data_cond = ((len(crow_uniq.values) == 1) and (np.isnan(crow_uniq.iat[0])))
                newc_ds.at[cinx] = np.nan if nv_data_cond else np.nanmax(crow)
            self.ldf[c] = newc_ds.copy()

        self.ldf = self.ldf[res_cols].sort_values(by=['dt'], ascending=False, ignore_index=True)

        ## Cr (S) 와 Creatinine 병합

        ldf_cols = list(self.ldf.columns)
        cr_list = ['Cr (S)', 'Creatinine']
        if ('Cr (S)' in ldf_cols) and ('Creatinine' in ldf_cols):
            for linx, lrow in self.ldf[cr_list].iterrows():
                cr_vals = []
                for cr in cr_list:
                    if type(lrow[cr]) == float:
                        cr_vals.append(lrow[cr])
                        continue
                    elif type(lrow[cr]) == str:
                        if lrow[cr] in ('', '-', '.'):
                            cr_vals.append(np.nan)
                        else:
                            cr_vals.append(float(lrow[cr].replace('<', '').replace('>', '')))
                self.ldf.at[linx, 'Cr (S)'] = np.nanmax(cr_vals)
        elif ('Cr (S)' not in ldf_cols) and ('Creatinine' in ldf_cols):
            self.ldf['Cr (S)'] = np.nan
            for linx, lrow in self.ldf[cr_list].iterrows():
                cr_list = ['Creatinine', ]
                cr_vals = []
                for cr in cr_list:
                    if type(lrow[cr]) == float:
                        cr_vals.append(lrow[cr])
                        continue
                    elif type(lrow[cr]) == str:
                        if lrow[cr] in ('', '-', '.'):
                            cr_vals.append(np.nan)
                        else:
                            cr_vals.append(float(lrow[cr].replace('<', '').replace('>', '')))
                self.ldf.at[linx, 'Cr (S)'] = np.nanmax(cr_vals)

        return self.ldf

    @staticmethod
    def get_reduced_sentence(sentence):
        if type(sentence) != str:
            print('인풋값이 str이 아닙니다. 원래 문장을 반환합니다.')
            return sentence
        prev_s = sentence
        while True:
            cur_s = prev_s.replace('  ', '')
            if cur_s == prev_s:
                return cur_s
            prev_s = cur_s

    def get_parsed_ecg_result(self, ecg_result):
        # ecg_result = self.pt_dict['ecg']
        ecg_result_df = pd.DataFrame(columns=['date', 'ecg_result'])
        if type(ecg_result) != str: pass
        else:
            ecg_result_df = [(ecgstr.split(')\n')[0].replace(' ','').replace('\n','').replace('(',''), ')\n'.join(ecgstr.split(')\n')[1:]).replace('\n',' ').strip()) for ecgstr in ecg_result.split('작성과: ')[1:]]
            ecg_result_df = pd.DataFrame(ecg_result_df, columns=['date', 'ecg_result']).sort_values(['date'],ascending=False)
        return ecg_result_df

    def get_ecg_text(self):
        ecg_text_list = list()
        # for inx, row in self.pt_dict['ecg'].iterrows(): break
        for inx, row in self.pt_dict['ecg'].iterrows():
            ecg_text_frag = f"({row['date']})\n{row['ecg_result']}"
            ecg_text_list.append(ecg_text_frag)
        ecg_text = '\n'.join(ecg_text_list)
        return ecg_text

    def get_parsed_echocardiography_result(self, echo_result):
        # echo_result = self.pt_dict['echocardiography']
        echo_result_df = pd.DataFrame(columns=['date', 'echo_result'])
        if type(echo_result) != str: pass
        elif echo_result == '': pass
        else:
            echo_result_df = [(s.split('작성과: ')[-1].replace(' ', '').split(')\n')[0].split('(')[-1].replace('\n', '').replace('.', '-'),self.get_replaced_str_from_tups(target_str=s.split('Summary')[0],tups=[(' / ','/'),('\n',' '),(' .','.'),(' -','-'),(' %','%')]).strip()) for s in echo_result.split('Conclusions')[1:]]
            echo_result_df = pd.DataFrame(echo_result_df, columns=['date', 'echo_result']).sort_values(['date'], ascending=False)
        return echo_result_df

    def get_echocardiography_text(self):

        echo_text_list = list()
        # for inx, row in self.pt_dict['ecg'].iterrows(): break
        for inx, row in self.pt_dict['echocardiography'].iterrows():
            echo_text_frag = f"({row['date']})\n{row['echo_result']}"
            echo_text_list.append(echo_text_frag)
        echo_text = '\n'.join(echo_text_list)
        return echo_text


    def get_parsed_eeg_result(self, eeg_result):
        # echo_result = self.pt_dict['electroencephalography']
        # eeg_result = input()
        eeg_result_df = pd.DataFrame(columns=['date', 'eeg_result'])
        if type(eeg_result) != str: pass
        elif eeg_result=='': pass
        else:
            eeg_result_df = [(s.replace(' ', '').split(')\n')[0].split('(')[-1].replace('\n', '').replace('.', '-'), self.get_replaced_str_from_tups(target_str=s.split('소견\n')[-1].split('작성자\n')[0], tups=[(' / ', '/'), ('\n', ' '), (' .', '.'), (' -', '-'), (' %', '%'), ('    ', ' '), ('   ', ' '), ('  ', ' ')]).strip()) for s in eeg_result.split('작성과: ')[1:]]
            eeg_result_df = pd.DataFrame(eeg_result_df, columns=['date', 'eeg_result']).sort_values(['date'], ascending=False)
        return eeg_result_df

    def get_eeg_text(self):

        eeg_text_list = list()
        for inx, row in self.pt_dict['electroencephalography'].iterrows():
            eeg_text_frag = f"({row['date']})\n{row['eeg_result']}"
            eeg_text_list.append(eeg_text_frag)
        eeg_text = '\n'.join(eeg_text_list)
        return eeg_text

    def input_recording(self, key, value):
        if key=='lab':
            pass
        else:
            pass
        f_path = f''
        return None

    def get_pt_hx_df(self, hx_str, type_filt=''):
        # hx_str = self.pt_hx_raw

        hx_parsing_list = [hx_str]

        # for htype in self.hx_type_tups: break
        # for htype in hx_type_tups: break
        for htype in self.hx_type_tups:
            new_parsing_list = list()
            mid_parsing_list = list()
            # for hsfrag in self.hx_parsing_list: break
            for hsfrag in hx_parsing_list:
                hsfrag_split = list()
                for hfinx, hf in enumerate(hsfrag.split(htype+'\n')):
                    if hfinx==0:
                        if hf=='':pass
                        else:
                            hsfrag_split.append(hf)
                    else: hsfrag_split.append(htype+'\n'+hf)
                mid_parsing_list+= hsfrag_split

            for hsfrag in mid_parsing_list:
                hsfrag_split = list()
                for hfinx, hf in enumerate(hsfrag.split(htype+' ')):
                    if hfinx==0:
                        if hf=='':pass
                        else:
                            hsfrag_split.append(hf)
                    else: hsfrag_split.append(htype+'\n'+hf)
                new_parsing_list+= hsfrag_split


            hx_parsing_list = new_parsing_list

        final_parsing_list = list()
        # s = hx_parsing_list[2]
        # for s in hx_parsing_list: break
        for s in hx_parsing_list:
            s_type = s.split('\n')[0]
            s_dep_date = s.split(s_type+'\n')[1].replace('작성과: ','').replace(' ','').split(')\n')[0]
            s_dep = "(".join(s_dep_date.split('(')[:-1])
            s_date = s_dep_date.split('(')[-1].replace('\n','')
            s_text = s


            q_dep = ''
            q_doc = ''
            r_dep = ''
            r_doc = ''
            # st.session_state['monitor'] = s
            cslt_q_split = s.split('의뢰진료과\n')
            if len(cslt_q_split) > 1:
                q_dep = cslt_q_split[-1].split('의뢰의사\n')[0].replace('\n', '').strip()
                q_doc = cslt_q_split[-1].split('의뢰의사\n')[-1].split('수신진료과\n')[0].replace('\n', '').strip()

            cslt_r_split = s.split('수신진료과\n')

            if len(cslt_r_split)>1:
                r_dep = cslt_r_split[-1].split('수신인\n')[0].replace('\n','').strip()
                r_doc = cslt_r_split[-1].split('수신인\n')[-1].split('응급여부:')[0].replace('\n','').strip()

            w_doc = ''
            if ('작성자\n' in s) or ('\n작성자' in s):
                writer_cand_s = s.split('작성자\n')[-1].split('\n작성자')[-1]
                w_doc = writer_cand_s.replace('\n','').replace(' ','').split(s_date)[0].strip()


            final_parsing_list.append({'type':s_type, 'department':s_dep, 'date':s_date,'writer': w_doc, 'cslt_q_dep':q_dep, 'cslt_q_doc':q_doc, 'cslt_r_dep':r_dep, 'cslt_r_doc':r_doc, 'text':s_text})


        # fdf = pd.DataFrame(final_parsing_list).sort_values(['date'], ascending=True, ignore_index=True)
        fdf = pd.DataFrame(final_parsing_list).reset_index(drop=True)


        # type_filt='타과회신'
        if type_filt=='': return fdf
        else: return fdf[fdf['type']==type_filt].reset_index(drop=True)


    def get_replaced_str_from_tups(self, target_str, tups):
        for tup in tups:
            target_str=target_str.replace(tup[0], tup[1])
        return target_str

    def patient_state_viewer_analysis(self, included_prx_sig='/Y'):
        self.ps_viewer_df = list()
        # st.session_state['memo'] = str(self.order_df['Acting'].iloc[16])
        for inx, row in self.order_df.iterrows():
            if (included_prx_sig in row['Acting']) and len(re.findall(r'\([A-Za-z]*\)',row['처방지시']))>0:
                drug_str = re.findall(r'\([A-Za-z]*\)',row['처방지시'])[0][1:-1]
                if drug_str in ('ASAP',):
                    continue
                else:
                    self.ps_viewer_df.append({'date':row['date'], 'drug':drug_str, 'value':1})
        self.ps_viewer_df = pd.DataFrame(self.ps_viewer_df)
        # st.session_state['memo'] = str(self.drug_orders_df)
        if len(self.ps_viewer_df)>0:
            self.ps_viewer_df = self.ps_viewer_df.pivot_table(index=['drug'], columns=['date'], values=['value'], aggfunc=np.nanmax)
        st.session_state['ps_viewer'] = self.ps_viewer_df.copy()

    def parse_order_record(self, order_str):
        # order_str = input().strip()
        # order_str = self.pt_dict['order']
        # order_str='j'
        # ''.split('\n')
        if order_str=='': return self.order_df
        raw_order_str_list = order_str.split('\n')
        # order_str[:200]
        # for inx, row in self.order_df.iterrows(): break
        # st.session_state['memo'] = f"{raw_order_str_list[0]}//{raw_order_str_list[1]}"

        parsed_order_list = list()
        for row in raw_order_str_list:
            if '\t' not in row: continue
            parsed_order_list.append(dict(list(zip(self.raw_order_cols, row.split('\t')))))

        if (len(parsed_order_list)==0):
            return self.order_df
        elif (len(parsed_order_list)==1):
            if len(list(parsed_order_list[0].keys()))==1:
                if (list(parsed_order_list[0].keys())[-1])=='처방지시':
                    self.order_df = pd.DataFrame(columns=self.raw_order_cols)
                    return self.order_df
        else: pass
        # self.result_order_cols
        self.order_df = pd.DataFrame(parsed_order_list)

        for c in ('date', 'time', 'D/C', '보류', '반납'):
            self.order_df[c] = ''
        # row = self.order_df.iloc[5]
        for inx, row in self.order_df.iterrows():

            if (type(row['발행의']) != str) or (type(row['발행처']) != str): continue
            else:
                if len(row['발행의'].split(' ')) > 2:
                    self.order_df.at[inx, '처방지시'] = row['처방지시'].strip()
                    self.order_df.at[inx, 'date'] = row['발행의'].split(' ')[-2]
                    self.order_df.at[inx, 'time'] = row['발행의'].split(' ')[-1]
                elif len(row['발행처'].split(' ')) > 2:
                    self.order_df.at[inx, '처방지시'] = row['비고'].strip()
                    self.order_df.at[inx, 'date'] = row['발행처'].split(' ')[-2]
                    self.order_df.at[inx, 'time'] = row['발행처'].split(' ')[-1]
                else:
                    continue
                for order_state in ('D/C', '보류', '반납'):
                    self.order_df.at[inx, order_state] = order_state in self.order_df.at[inx, '처방지시']

        self.order_df = self.order_df[self.result_order_cols]
        # self.order_df.columns
        return self.order_df

    def parse_patient_history(self, hx_df, cont_type=''):
        # self.pt_hx_df
        # hx_df = self.pt_hx_df.copy()
        # self.pt_hx_df =
        parsed_str = ''
        if cont_type == 'history':
            hc_text = ''
            csltreply_ui_text = ''
            csltreply_assess_text = ''
            replace_tups = [(' - ', '-'), (' / ', '/'), ('s/p\n', 's/p '), ('\n.', '.')]

            if len(hx_df[hx_df['type']=='입원경과'])>0:

            ## 입원경과에서 추출

                hc_text_raw = hx_df[hx_df['type']=='입원경과'].iloc[0]['text']
                try:hc_text = hc_text_raw.split('Assessment & Plan\n')[1].split('Assessment\n')[-1].split('Plan\n')[0].split('투여\n사유')[0]
                except:pass
                hc_text = self.get_replaced_str_from_tups(target_str=hc_text, tups=replace_tups)
                for m in range(1,10): hc_text = hc_text.replace(f'-0\n{m}-',f'-0{m}-')
                for m in range(1, 3): hc_text = hc_text.replace(f'-1\n{m}-', f'-1{m}-')

            ## 타과회신(기저질환항목)에서 추출
            # st.session_state['monitor'] = f"Drug : {self.pt_dict['drug']}\nPedi : {self.pt_dict['pedi']}"
            # st.session_state['monitor'] = f"Drug : {self.pt_dict['pedi']}"
            drug_cslt_dep = self.drug_consult_dict[self.pt_dict['drug']] if not self.pt_dict['pedi'] else '소아청소년과'

            if len(hx_df[hx_df['type'] == '타과회신']) > 0:
                if drug_cslt_dep == '':
                    csltreply_text_raw = hx_df[hx_df['type'] == '타과회신'].iloc[0]['text']
                else:
                    try:csltreply_text_raw = hx_df[(hx_df['type'] == '타과회신') & (hx_df['department'] == drug_cslt_dep)].iloc[0]['text']
                    except: csltreply_text_raw = hx_df[hx_df['type'] == '타과회신'].iloc[0]['text']
                try:csltreply_ui_text = csltreply_text_raw.split('회신내용\n')[-1].split('기저\n질환\n')[1].split('감염증(원발\n부위)')[0].split('투여\n사유')[0]
                except:
                    try:csltreply_ui_text = csltreply_text_raw.split('회신내용\n')[-1].split('Underlying\nillness\n')[1].split('감염증(원발\n부위)')[0].split('투여\n사유')[0]
                    except: pass
                csltreply_ui_text = self.get_replaced_str_from_tups(target_str=csltreply_ui_text, tups=replace_tups)

                ## 타과회신(Assessment & Plan)에서 추출

                try:csltreply_assess_text = csltreply_text_raw.split('회신내용\n')[-1].split('Assessment\n')[1].split('Plan\n')[0].split('Opinions or Recommendations\n')[0]
                except:
                    try:csltreply_assess_text = csltreply_text_raw.split('회신내용\n')[-1].split('Underlying\nillness\n')[1].split('감염증(원발\n부위)')[0].split('투여\n사유')[0]
                    except:
                        pass
                csltreply_assess_text = self.get_replaced_str_from_tups(target_str=csltreply_assess_text, tups=replace_tups)

            ## Pt_Hx 후보 통합문구

            hc_parsed_str = f"####### 최근입원경과Hx #######\n\n{hc_text}" if hc_text!='' else ''
            connector_str = "\n\n" if hc_parsed_str!='' else ""
            csltreply_ui_parsed_text = connector_str + f"####### 타과회신UI_Hx #######\n\n{csltreply_ui_text}" if csltreply_ui_text!='' else ''
            connector_str = "\n\n" if (hc_parsed_str != '') or (csltreply_ui_parsed_text != '') else ""
            csltreply_assess_parsed_text = connector_str + f"####### 타과회신Assessment_Hx #######\n\n{csltreply_assess_text}" if csltreply_assess_text!='' else ''

            parsed_str = f'{hc_parsed_str}{csltreply_ui_parsed_text}{csltreply_assess_parsed_text}'

        elif cont_type == 'hemodialysis':
            parsed_str = cont_type + " parsed"
        elif cont_type == 'consult':
            drug_cslt_dep = self.drug_consult_dict[self.pt_dict['drug']] if not self.pt_dict['pedi'] else '소아청소년과'

            parsed_str = ''
            if drug_cslt_dep=='':pass
            else:
                # hx_df = fdf
                cslt_df = hx_df[(hx_df['type'] == '타과의뢰')|(hx_df['type'] == '타과회신')].reset_index(drop=True)
                cslt_q_df = cslt_df[(cslt_df['type'] == '타과의뢰')]
                cslt_r_df = cslt_df[(cslt_df['type'] == '타과회신')]
                if len(cslt_q_df)==0:
                    parsed_str='(-)'
                    return parsed_str
                last_cslt_q_inx = cslt_q_df.iloc[-1].name

                cslt_pair_list = list()
                # csltinx = 1
                # csltrow = cslt_df.iloc[csltinx]
                # for csltinx, csltrow in cslt_df.iterrows(): break
                # st.session_state['monitor'] = cslt_df[['type','text']]
                for csltinx, csltrow in cslt_df.iterrows():
                    # if csltinx==3: break
                    # st.session_state['monitor'] = csltrow['cslt_r_dep']
                    # st.session_state['monitor'] = csltrow
                    # if csltinx==1: break
                    if csltrow['type']=='타과회신': continue
                    # elif (csltrow['type']=='타과의뢰') and (drug_cslt_dep not in csltrow['cslt_r_dep']): continue
                    else: pass

                    if csltinx != last_cslt_q_inx:
                        next_q_inx = cslt_q_df[cslt_q_df.index > csltinx].iloc[0].name
                    else: next_q_inx = np.inf

                    r_cands = cslt_df[(cslt_df.index>csltinx) & (cslt_df.index<next_q_inx) & (cslt_df['type']=='타과회신')]
                    if len(r_cands)==0:
                        continue
                    elif len(r_cands)>=2:
                        print(f'타과회신이 1개 이상 입니다. 확인해주세요')
                        print(f'[해당의뢰]\n{csltrow}')
                        for rcinx, rcval in r_cands.iterrows():
                            print(rcval)
                        # raise ValueError
                    else: pass

                    r_cand_row = r_cands.iloc[-1]

                    ## consult 과 매칭 로직 deprecated (에러발생으로_사용자가 필요한 과 COnsult 잘 넣는 것으로 !)
                    # infomatch_cond = ((r_cand_row['department'] in csltrow['cslt_r_dep']) or ((csltrow['cslt_r_dep'] in ('심장혈관센터', '순환기내과', '순환기내과(심장혈관센터)', '신경과', '신경과(뇌신경센터)')) and (r_cand_row['department'] in ('심장혈관센터', '순환기내과', '순환기내과(심장혈관센터)', '신경과', '신경과(뇌신경센터)'))))
                    # if infomatch_cond:
                    #     cslt_pair_list.append({'cslt_inx':csltinx, 'cslt_q_row':csltrow, 'cslt_r_row':r_cand_row})
                    # else:
                    #     print('의뢰, 회신 정보가 일치하지 않습니다. 확인해주세요')
                    #     print(csltrow)
                    #     print(r_cand_row)
                    #     raise ValueError

                    ## 의뢰텍스트 파싱
                    # st.session_state['monitor'] = csltrow['text']
                    cslt_q_text_raw = csltrow['text']
                    text_for_abx_exam_check = cslt_q_text_raw.replace('\n','').replace(' ','')
                    if (drug_cslt_dep=='감염내과') and ("[오더발행]" in text_for_abx_exam_check) and ("[동정결과]" in text_for_abx_exam_check) and ("[중간보고]" in text_for_abx_exam_check):

                        before_abx_exam_check = cslt_q_text_raw.split('[오더발행]')[0]
                        after_abx_exam_check = '[중간보고]'+cslt_q_text_raw.split('[중간보고]')[-1]

                        abx_exam_check = cslt_q_text_raw.split('[오더발행]')[-1].split('[동정결과]')[-1].split('[동정\n결과]')[-1].split('* 항생제\n감수성\n결과가\n있는\n경우')[0].strip()
                        abx_exam_check = self.get_replaced_str_from_tups(target_str=abx_exam_check, tups=[('\n년', '년'), ('\n월', '월'), ('\n일', '일'), ('\n', ' ')])

                        cslt_q_text_raw = before_abx_exam_check+abx_exam_check+after_abx_exam_check

                    cslt_q_text = cslt_q_text_raw
                    replace_tups = [(' - ', '-'), (' / ', '/'), ('s/p\n', 's/p '), ('\n.', '.'), ('\n', ' '), ('  ', '\n')]
                    elimlist_dict = {0:['의뢰내용\n', '제한항생제 투여 사유 : ', '안녕하십니까.','안녕하십니까,', '안녕하십니까', '선택제한항생제', '*제한항생제\n투여\n사유:'],
                                     1:['작성자\n', '\n작성자','Opinions or Recommendations\n'],
                                           }
                    for eliminx, elimlist in elimlist_dict.items():
                        for elimmargin in elimlist:
                            cslt_q_text = cslt_q_text.split(elimmargin)[eliminx-1]


                    ## Consult 파싱 이전코드
                    # cslt_q_text = ''
                    # replace_tups = [(' - ', '-'), (' / ', '/'), ('s/p\n', 's/p '), ('\n.', '.'), ('\n', ' '),
                    #                 ('  ', '\n')]
                    #
                    # try:cslt_q_text = cslt_q_text_raw.split('의뢰내용\n')[-1].cslt_q_text_raw.split('투여\n사유:\n')[-1].split('안녕하십니까.\n')[-1].split('안녕하십니까\n')[-1].split('작성자\n')[0].split('\n작성자')[0].split('의견\n')[1]
                    # except:
                    #     try:cslt_q_text = cslt_q_text_raw.split('의뢰내용\n')[-1].cslt_q_text_raw.split('투여\n사유:\n')[-1].split('안녕하십니까.\n')[-1].split('안녕하십니까\n')[-1].split('작성자\n')[0].split('\n작성자')[0].split('Opinions or Recommendations\n')[1]
                    #     except:cslt_q_text = cslt_q_text_raw.split('의뢰내용\n')[-1].cslt_q_text_raw.split('투여\n사유:\n')[-1].split('안녕하십니까.\n')[-1].split('안녕하십니까\n')[-1].split('작성자\n')[0].split('\n작성자')[0].split('감사합니다.')[0].split('감사합니다')[0]

                    cslt_q_text = self.get_replaced_str_from_tups(target_str=cslt_q_text, tups=replace_tups).strip()

                    ## 회신텍스트 파싱

                    cslt_r_text_raw = r_cand_row['text']
                    cslt_r_text = cslt_r_text_raw
                    replace_tups = [(' - ', '-'), (' / ', '/'), ('s/p\n', 's/p '), ('\n.', '.'), ('\n', ' '), ('  ', '\n')]
                    elimlist_dict = {0: ['회신내용\n', '의견\n', 'Opinions or Recommendations\n'],
                                     1: ['\n작성자', '작성자\n'],
                                     }
                    for eliminx, elimlist in elimlist_dict.items():
                        for elimmargin in elimlist:
                            cslt_r_text = cslt_r_text.split(elimmargin)[eliminx - 1]


                    # try:cslt_r_text = cslt_r_text_raw.split('회신내용\n')[-1].split('작성자\n')[0].split('\n작성자')[0].split('의견\n')[1]
                    # except:
                    #     try:cslt_r_text = cslt_r_text_raw.split('회신내용\n')[-1].split('작성자\n')[0].split('\n작성자')[0].split('Opinions or Recommendations\n')[1]
                    #     except: cslt_r_text = cslt_r_text_raw.split('회신내용\n')[-1].split('작성자\n')[0].split('\n작성자')[0]
                    cslt_r_text = self.get_replaced_str_from_tups(target_str=cslt_r_text, tups=replace_tups).strip()

                    additional_text = f"({csltrow['date']})\n{cslt_q_text} -> {cslt_r_text}\n"

                    parsed_str += additional_text


                # cslt_reply_df = hx_df[(hx_df['type'] == '타과회신') & (hx_df['department'] == drug_cslt_dep)]
                #
                # for csltinx, csltrow in cslt_reply_df.iterrows():
                # # for csltinx, csltrow in cslt_df.iterrows():break
                #     cslt_text_raw = csltrow['text']
                #     cslt_text = ''
                #     replace_tups = [(' - ', '-'), (' / ', '/'), ('s/p\n', 's/p '), ('\n.', '.'), ('\n', ' '), ('  ', '\n')]
                #     try:cslt_text = cslt_text_raw.split('회신내용\n')[-1].split('작성자\n')[0].split('\n작성자')[0].split('의견\n')[1]
                #     except:
                #         try: cslt_text = cslt_text_raw.split('회신내용\n')[-1].split('작성자\n')[0].split('\n작성자')[0].split('Opinions or Recommendations\n')[1]
                #         except: pass
                #     cslt_text = self.get_replaced_str_from_tups(target_str=cslt_text, tups=replace_tups)
                #     additional_text = f"({csltrow['date']})\n{cslt_text}"
                #     parsed_str+=additional_text



        else: parsed_str = 'empty'
        return parsed_str.strip()

    def get_lab_text(self, drug):
        # drug = self.pt_dict['drug']
        # drug = 'VCM'
        self.drug_lablist_dict = {'VCM': {'WBC(seg%)/ANC': ['WBC', 'Seg.neut.', 'ANC'],
                                          'BUN/Cr': ['BUN', 'Cr (S)'],
                                          'GFR': ['eGFR-MDRD', 'eGFR-CKD-EPI', 'eGFR-Schwartz(소아)', '나이'],
                                          'CRP': ['CRP', ],
                                          'Alb': ['Albumin']
                                          },
                                  "DGX": {'BUN/Cr': ['BUN', 'Cr (S)'],
                                          'Ca/K': ['Ca, total', 'K'],
                                          'Alb': ['Albumin']
                                          },
                                  'AMK': {'WBC(seg%)/ANC': ['WBC', 'Seg.neut.', 'ANC'],
                                          'BUN/Cr': ['BUN', 'Cr (S)'],
                                          'GFR': ['eGFR-MDRD', 'eGFR-CKD-EPI', 'eGFR-Schwartz(소아)', '나이'],
                                          'CRP': ['CRP', ],
                                          'Alb': ['Albumin']
                                          },
                                  'GTM': {'WBC(seg%)/ANC': ['WBC', 'Seg.neut.', 'ANC'],
                                          'BUN/Cr': ['BUN', 'Cr (S)'],
                                          'GFR': ['eGFR-MDRD', 'eGFR-CKD-EPI', 'eGFR-Schwartz(소아)', '나이'],
                                          'CRP': ['CRP', ],
                                          'Alb': ['Albumin']
                                          },
                                  'VPA': {'PLT/PT/aPTT': ['Platelet', 'PT %', 'aPTT'],
                                          'BUN/Cr': ['BUN', 'Cr (S)'],
                                          'T.bil/AST/ALT': ['T.B', 'AST', 'ALT'],
                                          'Ammonia': ['Ammo', ],
                                          'Alb': ['Albumin']
                                          },

                                  }

        self.lablist_dict = self.drug_lablist_dict[drug]


        lab_text = ""
        labtups = tuple(self.drug_lablist_dict[drug].keys())
        self.labres_dict = dict([(c, '') for c in labtups])
        self.labrescount_dict = dict([(c, 0) for c in labtups])

        uniq_date = [d for d in self.ldf['date'].unique() if (d >= self.prev_date)]
        uniq_date.sort(reverse=True)
        # for ud in uniq_date: break

        # self.ldf.columns
        # for ud in uniq_date: break
        # ud = '2023-10-20'
        for ud in uniq_date:
            ldf_frag = self.ldf[self.ldf['date'] == ud].copy()
            # for lab_key, lab_list in self.lablist_dict.items(): break
            for lab_key, lab_list in self.lablist_dict.items():
                # if lab_key=='Ammonia': break
                lab_list_except_age = list(set(lab_list) - {'나이', })
                try:
                    sub_ldf_frag = ldf_frag[['dt'] + lab_list_except_age].copy()
                    sub_ldf_frag[lab_list_except_age] = sub_ldf_frag[lab_list_except_age].replace('', np.nan).replace('-', np.nan).replace('.', np.nan).applymap(lambda x: float(x.replace('<', '').replace('>', '')) if (type(x) == str) else x).replace('-', np.nan)
                    non_null_frag = sub_ldf_frag[~sub_ldf_frag[lab_list_except_age].isnull().all(axis=1)].copy()
                except:
                    self.labres_dict[lab_key]='.'
                    non_null_frag = pd.DataFrame(columns=['dt'])
                    continue


                non_null_frag['나이'] = self.pt_dict['age']
                if (len(non_null_frag) == 0) or (self.labrescount_dict[lab_key] > 4):
                    pass
                else:
                    addtext = parse_daily_notnull_labdf_into_addtext(lab_date=ud, lab_key=lab_key, not_null_df=non_null_frag)
                    self.labres_dict[lab_key] += addtext
                    self.labrescount_dict[lab_key] += 1


        for lk, ftxt in self.labres_dict.items():
            if ftxt != '':
                self.labres_dict[lk] = ftxt[:-2].strip()
            if lk in ('Alb', 'Ammonia'):
                self.labres_dict[lk] = ftxt[:-2].split(" <-")[0].strip()
            lab_text += f"{lk}: {self.labres_dict[lk]}\n"
        lab_text = lab_text + "\n"

        return lab_text

    def get_drug_administration_hx(self, drug):
        # drug = self.pt_dict['drug']
        # self.get_drug_administration_hx_full_text(drug)

        # adf = self.order_df[(self.order_df['D/C']==False)].reset_index(drop=True)
        adf = self.order_df.reset_index(drop=True)
        # for inx, row in adf.iterrows(): break
        inx_list = set()
        for inx, row in adf.iterrows():
             for drugfn in self.drug_fullname_dict[drug]:
                 if drugfn in row['처방지시'].upper():
                     inx_list.add(inx)

        dodf = adf[adf.index.isin(list(inx_list))].reset_index(drop=True)

        dodf['약물용량'] = ''
        dodf['투여방식'] = ''
        dodf['투여방식추가'] = ''
        dodf['투여방식상세'] = ''
        dodf['투여간격'] = ''
        dodf['투여간격추가'] = ''
        dodf['투여날짜'] = ''
        dodf['투여시간list'] = ''

        if len(dodf)==0:
            return dodf

        ## 약어 참고
        # MIV : Mix Intra Venous (혼합정맥주사) / PLT : PER L-Tube (L-tube로 주입) / IVS : IV side push (수액주입경로를 통한 정맥주사)
        # q24h : 24시간마다 1회 투여 / q12h : 12시간마다 1회 투여
        # qd : 하루 1회 투여 / bid : 하루 2회 투여 / tid : 하루 3회 투여 / qid : 하루 4회 투여

        # dodf.to_csv(f'{project_dir}/result/drug_order.csv', encoding='utf-8-sig', index=False)
        # drugfn = 'VANCOMYCIN'
        if drug=='VCM': dodf = dodf[dodf['처방지시'].map(lambda x: False if ("VANCOMYCIN TDM" in x.upper()) or ("VANCOMYCIN 농도 [SERUM]" in x.upper()) or (x[:4] in ('* PO','* IV'))else True)].reset_index(drop=True)
        elif drug == 'DGX':dodf = dodf[dodf['처방지시'].map(lambda x: False if ("DIGOXIN TDM" in x.upper()) or ("DIGOXIN 농도 [SERUM]" in x.upper()) or (x[:4] in ('* PO','* IV'))else True)].reset_index(drop=True)
        elif drug == 'AMK':dodf = dodf[dodf['처방지시'].map(lambda x: False if ("AMIKACIN TDM" in x.upper()) or ("AMIKACIN 농도 [SERUM]" in x.upper()) or (x[:4] in ('* PO','* IV'))else True)].reset_index(drop=True)
        elif drug == 'GTM':dodf = dodf[dodf['처방지시'].map(lambda x: False if ("GENTAMICIN TDM" in x.upper()) or ("GENTAMICIN 농도 [SERUM]" in x.upper()) or (x[:4] in ('* PO', '* IV')) else True)].reset_index(drop=True)
        elif drug == 'VPA':dodf = dodf[dodf['처방지시'].map(lambda x: False if ("VALPROATE TDM" in x.upper()) or ("VALPROATE 농도 [SERUM]" in x.upper()) or (x[:4] in ('* PO','* IV'))else True)].reset_index(drop=True)
        else:dodf = dodf[dodf['처방지시'].map(lambda x: False if (f"{self.drug_fullname_dict[drug][0]} TDM" in x.upper()) or (f"{self.drug_fullname_dict[drug][0]} 농도 [SERUM]" in x.upper()) or (x[:4] in ('* PO','* IV'))else True)].reset_index(drop=True)


        dodf = dodf[dodf['Acting'].map(lambda x:x.strip())!=''].reset_index(drop=True)

        # for doinx, dorow in dodf.iterrows(): break

        admin_path_dict = {"[MIV]" : "IV", "[PLT]": "PO", "[P.O]":"PO", "[IVS]":"IV"}
        # doinx = 0
        # dorow = dodf.iloc[doinx]
        # dodf['처방지시']
        for doinx, dorow in dodf.iterrows():
            doinfo_str = dorow['처방지시']

            ## 약물별 문구 파싱

            if drug=='VCM': ddparsing_list = [dconc.strip() for dconc in (re.findall(r' [\d]+g ', doinfo_str) + re.findall(r' [\d]+mg ', doinfo_str))]
            elif drug == 'DGX':
                tab_list = [float(dconc.replace('mg','').strip()) for dconc in re.findall(r' [\d]*\.?[\d]+mg ', doinfo_str)]
                if len(tab_list)==0: continue
                tab_mass = tab_list[-1]
                adm_mass_list = [float(dconc.replace('tab','').replace('amp','').strip()) for dconc in re.findall(r' [\d]*\.?[\d]+ tab', doinfo_str) + re.findall(r' [\d]*\.?[\d]+ amp', doinfo_str)]
                if (len(adm_mass_list)==0) and (len(tab_list)>1): adm_mass = 1
                else: adm_mass = adm_mass_list[-1]
                # adm_mass = [float(dconc.replace('tab', '').replace('amp', '').strip()) for dconc in re.findall(r' [\d]*\.?[\d]+ tab', doinfo_str) + re.findall(r' [\d]*\.?[\d]+ amp', doinfo_str)][-1]
                ddparsing_list = [f"{tab_mass}mg", f"{tab_mass*adm_mass}mg"]
            elif drug == 'AMK': ddparsing_list = [dconc.strip() for dconc in (re.findall(r' [\d]+g ', doinfo_str) + re.findall(r' [\d]+mg ', doinfo_str))]
            elif drug == 'GTM': ddparsing_list = [dconc.strip() for dconc in (re.findall(r' [\d]+g ', doinfo_str) + re.findall(r' [\d]+mg ', doinfo_str))]
            elif drug == 'VPA': ddparsing_list = [dconc.strip() for dconc in (re.findall(r' [\d]+g ', doinfo_str) + re.findall(r' [\d]+mg ', doinfo_str))]
            else: ddparsing_list = [dconc.strip() for dconc in (re.findall(r' [\d]+g ', doinfo_str) + re.findall(r' [\d]+mg ', doinfo_str))]

            if len(ddparsing_list)==0:
                dodf.at[doinx, '투여시간list'] = list()
                continue
            dodf.at[doinx, '약물용량'] = ddparsing_list[-1]
            pharmexam_list = re.findall(r'[\d][\d][\d][\d]-[\d][\d]-[\d][\d]', dorow['약국/검사']) + re.findall(r'[\d][\d][\d][\d]-[\d][\d]-[\d][\d]',dorow['Acting'])
            if len(pharmexam_list)==0: dodf.at[doinx, '투여날짜'] = dorow['date']
            else: dodf.at[doinx, '투여날짜'] = max(pharmexam_list)
                # for dapk, dapv in admin_path_dict.items(): break
            for dapk, dapv in admin_path_dict.items():
                if dapk in doinfo_str:
                    dodf.at[doinx, '투여방식'] = admin_path_dict[dapk]
                    dodf.at[doinx, '투여방식상세'] = dapk
                    dodf.at[doinx, '투여간격'] = doinfo_str.split(dapk+' ')[-1].split(' ')[0]
                    break
                else:
                    continue
                    # dodf.iloc[6]
            if (drug=='VCM') and ((dodf.at[doinx, '투여방식'] == 'IV') and (dodf.at[doinx, '약물용량'].replace('mg','') >= '1400') and (dodf.at[doinx, '투여간격'] == 'x1')): dodf.at[doinx, '투여방식추가'] = ' Loading'
            elif (drug=='DGX') and ('POW' in doinfo_str): dodf.at[doinx, '투여간격추가'] = ' Powder'

            else: pass
            acting_list = dorow['Acting'].split(', ')
            y_acting_list = list()
            for al in acting_list:
                if al=='': continue
                else:
                    if al[-1].upper() in ('Y','Z','C'):
                        # al = y_acting_list[0]
                        al_split = al.split('/')
                        hr_min = al_split[0]
                        hr_min_split = hr_min.split(':')
                        hr_str = hr_min_split[0].split(' ')[-1]
                        min_str = hr_min_split[1]

                        if int(hr_str) > 12:
                            adm_hr = str(int(hr_str)-12)
                            am_pm = 'P'
                        elif int(hr_str) == 12:
                            adm_hr = 'MD'
                            am_pm = ''
                        elif (int(hr_str) <12) and (int(hr_str) >= 0):
                            adm_hr = str(int(hr_str))
                            am_pm = 'A'
                        adm_min = str(int(min_str))
                        adm_time = f'{adm_hr}{am_pm}{adm_min.zfill(2)}'
                        y_acting_list.append(adm_time)
            dodf.at[doinx, '투여시간list'] = y_acting_list

        return dodf

    def get_drug_administration_hx_full_text(self, drug):
        # drug = 'VCM'
        drug_full_name = self.drug_fullname_dict[drug][0]
        drug_full_name = drug_full_name[0].upper() + drug_full_name[1:].lower()
        # adm_df['투여시간list']
        # adm_df.columns
        # adm_df = dodf.copy()
        adm_df = self.get_drug_administration_hx(drug=drug)
        adm_df = adm_df[adm_df['투여시간list'].map(lambda x: len(x) > 0)].reset_index(drop=True)
        if len(adm_df)==0:
            full_adm_text = f"*이전 투약력\n\n*현 투약력"
            min_prev_adm_dtstr = '0000-01-01'
            return full_adm_text, min_prev_adm_dtstr

        adm_df = adm_df.sort_values(['투여날짜'], ascending=False, ignore_index=True)

        uniq_date_list = list(adm_df['투여날짜'].drop_duplicates())
        dup_date_list = [d for d in uniq_date_list if ((adm_df['투여날짜']==d).sum())>1]
        # d=dup_date_list[0]
        adm_df_cols = list(adm_df.columns)
        for d in dup_date_list:
            upper_df = adm_df[adm_df['투여날짜'] > d].copy()
            lower_df = adm_df[adm_df['투여날짜'] < d].copy()
            shuffle_df = adm_df[adm_df['투여날짜'] == d].copy()
            # shuffle_df['투여시간_min_num'] = shuffle_df['투여시간list'].map(lambda x: x[0][-2:])
            shuffle_df['투여시간_hr_num'] = shuffle_df['투여시간list'].map(lambda x: x[0][0])
            shuffle_df['투여시간_am_md_pm'] = shuffle_df['투여시간list'].map(lambda x:x[0][1])
            shuffle_df = shuffle_df.sort_values(['투여시간_am_md_pm', '투여시간_hr_num'], ascending=False)[adm_df_cols]
            adm_df = pd.concat([upper_df, shuffle_df, lower_df], ignore_index=True)



        # adm_df[['투여날짜', '약물용량', '투여시간list']]
        # adm_df[['투여날짜', '약물용량', '투여방식', '투여방식추가', '투여방식상세', '투여간격', '투여시간list']]

        adm_df['투약력문구_raw'] = adm_df.apply(lambda x: f"{drug_full_name} {x['약물용량']} {x['투여방식']}{x['투여방식추가']} {x['투여간격']}{x['투여간격추가']}", axis=1)

        ## VCM AUC 구할때 필요
        if len(adm_df)!=0:
            self.last_adm_info_row = adm_df.iloc[0]

        cur_adm = None
        cur_adm_dt_dict = dict()
        cur_adm_dt_list = list()
        cur_adm_dtstr = ''
        prev_adm = None
        prev_adm_dt_dict = dict()
        prev_adm_dt_list = list()
        prev_adm_dtstr = ''
        third_adm = None
        # inx=4
        # row = adm_df.iloc[inx]
        for inx, row in adm_df.iterrows():  # break
            # if inx==3: raise ValueError

            ## cur_adm 및 prev_adm 기준문구 결정

            if (cur_adm is None) and (row['투약력문구_raw'] != ''):
                cur_adm = row['투약력문구_raw']
            elif (cur_adm is not None) and (prev_adm is None):
                if (cur_adm != row['투약력문구_raw']) and (row['투약력문구_raw'] != ''):
                    prev_adm = row['투약력문구_raw']
            elif (cur_adm is not None) and (prev_adm is not None):
                if (cur_adm != row['투약력문구_raw']) and (prev_adm != row['투약력문구_raw']):
                    third_adm = row['투약력문구_raw']
                    break


            ## 문구가 결정되었을때 투여 datetime 문구 결정

            if (cur_adm is not None) and (prev_adm is None):
                if (cur_adm == row['투약력문구_raw']):
                    if row['투여날짜'] in cur_adm_dt_list:
                        cur_adm_dt_dict[row['투여날짜']] += f"/{'/'.join(row['투여시간list'])}"
                        # cur_adm_dtstr += '/'.join(row['투여시간list']) + ' '
                    else:
                        tdm_date_str = get_tdm_dateform(date_str=row['투여날짜'])
                        cur_adm_dt_dict[row['투여날짜']]=f"{tdm_date_str} {'/'.join(row['투여시간list'])}"
                        # cur_adm_dtstr = f"{tdm_date_str} {'/'.join(row['투여시간list'])} " + cur_adm_dtstr
                        # cur_adm_dt_list.append(row['투여날짜'])

            elif (prev_adm is not None) and (third_adm is None):
                if (prev_adm == row['투약력문구_raw']):

                    if (row['투여날짜'] in list(prev_adm_dt_dict.keys())):
                        prev_adm_dt_dict[row['투여날짜']] += f"/{'/'.join(row['투여시간list'])}"
                        # prev_adm_dtstr += '/'.join(row['투여시간list']) + ' '
                    else:
                        tdm_date_str = get_tdm_dateform(date_str=row['투여날짜'])
                        prev_adm_dt_dict[row['투여날짜']] = f"{tdm_date_str} {'/'.join(row['투여시간list'])}"
                        # prev_adm_dtstr = f"{tdm_date_str} {'/'.join(row['투여시간list'])} " + prev_adm_dtstr
                        # prev_adm_dt_list.append(row['투여날짜'])

        # cur_adm_dtstr = cur_adm_dtstr.strip()
        # prev_adm_dtstr = prev_adm_dtstr.strip()

        for v in list(cur_adm_dt_dict.values()): cur_adm_dtstr = f'{v}, {cur_adm_dtstr}' if cur_adm_dtstr!='' else v
        for v in list(prev_adm_dt_dict.values()): prev_adm_dtstr = f'{v}, {prev_adm_dtstr}' if prev_adm_dtstr != '' else v

        # 여기작업중 (max, min 투약날짜 및 시간 저장 -> Steady State 여부 판별위해)
        # cur_adm_dt_dict
        # cur_adm_date_list = list(cur_adm_dt_dict.keys())
        # min_cur_adm_date = min(cur_adm_date_list)
        # max_cur_adm_date = max(cur_adm_date_list)




        cur_adm_text = f"\n{cur_adm} ({cur_adm_dtstr})" if cur_adm is not None else '(-)'
        prev_adm_text = f"\n{prev_adm} ({prev_adm_dtstr})" if prev_adm is not None else '(-)'

        full_adm_text = f"*이전 투약력{prev_adm_text}\n\n*현 투약력{cur_adm_text}"

        min_prev_adm_dtstr = min(prev_adm_dt_list) if len(prev_adm_dt_list)>0 else '0000-01-01'
        return full_adm_text, min_prev_adm_dtstr

    def get_base_cr_gfr_text(self, prev_adm_date):
        cr_base_value = np.nan
        gfr_base_value = np.nan

        if len(self.pt_dict['lab'])==0:
            cr_base_date_txt=''
            gfr_base_date_txt=''
            return {"Cr": (cr_base_date_txt,cr_base_value), "GFR": (gfr_base_date_txt,gfr_base_value)}

        cr_base_df = self.pt_dict['lab'][['dt', 'date', 'Cr (S)']].copy()
        cr_base_df = cr_base_df.dropna().reset_index(drop=True)
        if prev_adm_date > cr_base_df['date'].min():
            base_lab_threshold_date = prev_adm_date
        else:
            base_lab_threshold_date = (datetime.strptime(cr_base_df['date'].min(), '%Y-%m-%d') + timedelta(days=1)).strftime('%Y-%m-%d')
        cr_base_df = cr_base_df[cr_base_df['date'] < base_lab_threshold_date].copy()
        cr_base_dt =  cr_base_df['dt'].max()
        cr_base_date = cr_base_dt.split('T')[0]

        cr_base_date_txt = get_tdm_dateform(date_str=cr_base_date)

        cr_base_value = cr_base_df[cr_base_df['dt'] == cr_base_dt].iloc[0]['Cr (S)']

        gfr_cols = [c for c in self.pt_dict['lab'].columns if ('GFR' in c.upper())]
        gfr_base_df = self.pt_dict['lab'][['dt', 'date'] + gfr_cols].copy()
        gfr_c = ''
        if self.pt_dict['age'] >= 18:
            if 'eGFR-MDRD' in gfr_cols:
                gfr_c = 'eGFR-MDRD'
            else:
                if 'eGFR-CKD-EPI' in gfr_cols:
                    gfr_c = 'eGFR-CKD-EPI'
                else:
                    gfr_c = 'eGFR-Cockcroft-Gault'
        else:
            gfr_c = 'eGFR-Schwartz(소아)'

        gfr_base_df = gfr_base_df[['dt', 'date', gfr_c]].copy()
        gfr_base_df = gfr_base_df.dropna().reset_index(drop=True)
        if prev_adm_date > gfr_base_df['date'].min():
            base_lab_threshold_date = prev_adm_date
        else:
            base_lab_threshold_date = (
                        datetime.strptime(gfr_base_df['date'].min(), '%Y-%m-%d') + timedelta(days=1)).strftime(
                '%Y-%m-%d')
        gfr_base_df = gfr_base_df[gfr_base_df['date'] < base_lab_threshold_date].copy()
        gfr_base_dt = gfr_base_df['dt'].max()
        gfr_base_date = gfr_base_dt.split('T')[0]


        gfr_base_date_txt = get_tdm_dateform(date_str=gfr_base_date)

        gfr_base_value = gfr_base_df[gfr_base_df['dt'] == gfr_base_dt].iloc[0][gfr_c]



        result_dict = {"Cr": (cr_base_date_txt,cr_base_value), "GFR": (gfr_base_date_txt,gfr_base_value)}
        return result_dict

    def get_concomitant_mdx_text(self, drug):
        # drug = self.pt_dict['drug']

        # dodf.to_csv(f'{project_dir}/result/concomitant_order.csv', encoding='utf-8-sig', index=False)

        conc_mdx_str = ''

        self.conc_mdx_dict = {'VCM': ['Furosemide', 'Piperacillin/Tazobactam', 'Penem', 'avir', 'ovir', 'cillin', 'mycin', 'ceft', 'tacrolimus', 'cefepime', 'xacin', 'rifampicin', 'rifampin','sulfamethoxazol', 'trimethoprim', 'cilastatin', 'Metronidazole','Refampin', 'colistin', 'fungin', 'gentamicin', 'mycin', 'fluconazole', 'amikacin', 'cycloserine', 'ethambutol', 'ampotericin B', 'conazole'],
                              'AMK': ['Furosemide', 'Piperacillin/Tazobactam', 'Penem', 'avir', 'ovir', 'cillin', 'mycin', 'ceft', 'tacrolimus', 'cefepime', 'xacin', 'rifampicin', 'rifampin','sulfamethoxazol', 'trimethoprim', 'cilastatin', 'Metronidazole','Refampin', 'colistin', 'fungin', 'gentamicin', 'mycin', 'fluconazole', 'amikacin', 'cycloserine', 'ethambutol', 'ampotericin B', 'conazole'],
                              'GTM': ['Furosemide', 'Piperacillin/Tazobactam', 'Penem', 'avir', 'ovir', 'cillin', 'mycin', 'ceft', 'tacrolimus', 'cefepime', 'xacin', 'rifampicin', 'rifampin', 'sulfamethoxazol', 'trimethoprim', 'cilastatin', 'Metronidazole', 'Refampin', 'colistin', 'fungin', 'gentamicin', 'mycin', 'fluconazole', 'amikacin', 'cycloserine', 'ethambutol', 'ampotericin B', 'conazole'],
                              'DGX': ['spironolactone', 'diltiazem','prolol', 'dipine', 'cyclosporine', 'norepinephrine', 'carbedilol', 'esmolol', 'clonazepam','captopril', 'Triamterene', 'hydrochlorothiazide', 'Amiodarone', 'dronedarone', 'itraconazole', 'macrolide', 'clarythromycin, erythromycin, tetracyclin', 'verapamil', 'rifampin', 'Sucralfate', 'Alprazolam', 'calcium'],
                              'VPA': ['levetiracetam'],
                              }
        cm_except_list = [d.lower() for d in self.drug_fullname_dict[drug]]
        cm_cand_list = self.conc_mdx_dict[drug]
        if len(self.order_df)==0:
            return conc_mdx_str
        dodf = self.order_df[(self.order_df['D/C'] == False)].reset_index(drop=True)
        check_date_list = list(dodf['date'].drop_duplicates().sort_values(ascending=False, ignore_index=True).iloc[:2])
        dodf = dodf[dodf['date'].isin(check_date_list)].copy()
        dodf = dodf[~dodf['Acting'].isna()].reset_index(drop=True)
        # dodf = dodf[dodf['Acting'].map(lambda x: ((x.strip()) != '') and ('/Y' in x.upper()))].reset_index(drop=True)
        # for inx, row in adf.iterrows(): break
        cm_dict = dict()
        # for inx, row in dodf.iterrows(): break
        for inx, row in dodf.iterrows():
            # cm_cand_frag = cm_cand_list[1]
            for cm_cand_frag in cm_cand_list:
                cm_cands_in_list = re.findall(fr"[\s(][\w]*{cm_cand_frag.upper()}[\w]*[\s)]", row['처방지시'].upper())
                if len(cm_cands_in_list) == 0: continue
                else:
                    cm_cand = self.get_replaced_str_from_tups(target_str=cm_cands_in_list[0], tups=[('(',''),(')',''),(' ','')])
                    cm_cand_lower = cm_cand.lower()
                    try: cm_dict[cm_cand_lower].add(inx)
                    except:
                        if cm_cand_lower in cm_except_list: continue
                        cm_dict[cm_cand_lower] = set()
                        cm_dict[cm_cand_lower].add(inx)

                    # inx_list.add(inx)
                    # cm_list.add(cm_cand.lower())

        conc_mdx_str = ', '.join(list(cm_dict.keys()))
        conc_mdx_str = conc_mdx_str.replace('mebapenem','meropenem')
        self.pt_dict['concomitant_medi'] = conc_mdx_str

        # dodf = dodf[dodf.index.isin(list(inx_list))].reset_index(drop=True)
        return self.pt_dict['concomitant_medi']

    def parse_vs_record(self, raw_vs):
        # self.raw_vs_str = raw_vs
        # raw_vs = self.raw_vs_str
        # print(raw_vs)
        # raw_vs = input()

        # drug = self.pt_dict['drug']
        # vdf = self.pt_dict['vs']

        raw_vs_str_list = [rv for rv in raw_vs.split('\n') if rv!='']
        if len(raw_vs_str_list)==0:
            self.vs_df = pd.DataFrame(columns=['SBP (mmHg)', 'DBP (mmHg)', 'PR (회/min)', 'BT (℃)', 'date'])
            return self.vs_df

        self.raw_vs_cols = [vsc.split('\t')[0] for vsc in raw_vs_str_list if vsc!='']

        parsed_vs_dict = dict()
        sc_cnt=0
        # for row in raw_vs_str_list: break
        for inx, row in enumerate(raw_vs_str_list):
            vs_split = row.split('\t')
            if inx==0:
                sc_cnt = len(vs_split)
            else:
                cc_cnt = len(vs_split)
                if len(vs_split)==0:pass
                else: vs_split+=['' for rcc in range(sc_cnt-cc_cnt)]


            parsed_vs_dict[vs_split[0]] = vs_split[1:]
            # print(vs_split[0], '/', len(vs_split))
        # self.result_order_cols
        self.vs_df = pd.DataFrame(parsed_vs_dict)
        self.vs_df['date'] = ''

        ed_prev = 5
        ed_last_date = self.pt_dict['tdm_date']
        ed_last_dt = datetime.strptime(ed_last_date,'%Y-%m-%d')
        # ed_inx = 0
        ed_win = int(len(self.vs_df)/ed_prev)
        for ed_inx in range(ed_prev):
            if ed_inx==ed_prev-1: ed_df = self.vs_df[ed_inx*ed_win:].copy()
            else:ed_df=self.vs_df[ed_inx*ed_win:(ed_inx+1)*ed_win].copy()
            for inx, row in ed_df.iterrows():
                self.vs_df.at[inx,'date'] = (ed_last_dt - timedelta(days=ed_prev-ed_inx-1)).strftime('%Y-%m-%d')
            # ed_inx+=1


        for c in list(self.vs_df.columns):
            if c in ('SBP (mmHg)', 'DBP (mmHg)', 'PR (회/min)', 'BT (℃)'): self.vs_df[c] = self.vs_df[c].map(lambda x: float(x) if x!='' else np.nan)
            else: pass

        return self.vs_df

    def get_vs_text(self, drug):
        # drug = self.pt_dict['drug']
        vs_text = ''
        # vs_df = pd.DataFrame(columns=['date','BT (℃)','SBP (mmHg)','DBP (mmHg)','PR (회/min)'])
        if drug in ('VCM', 'AMK', 'GTM'):
            vs_text_dict = {"*Max BT": [], }
            date_list = list(self.vs_df['date'].unique())
            date_list.sort(reverse=True)
            for vdate in date_list:
                vdf_frag = self.vs_df[self.vs_df['date'] == vdate].copy()
                tdmform_vdate = get_tdm_dateform(date_str=vdate)

                max_bt = np.nanmax(vdf_frag['BT (℃)'])
                bt_range_txt = f"{float(max_bt) if not np.isnan(max_bt) else ''}"

                vs_text_dict["*Max BT"].append(f"({tdmform_vdate}){bt_range_txt}")

            vs_text = "\n".join([f"{k}: {' <-'.join(v)}" for k, v in vs_text_dict.items()])

        elif drug=='DGX':
            # for vdate, vdf_frag in self.vs_df.groupby(by=['date']): break
            vs_text_dict = {"SBP/DBP": [], "HR": [],}
            date_list = list(self.vs_df['date'].unique())
            date_list.sort(reverse=True)
            # date_list = []
            for vdate in date_list:
                vdf_frag = self.vs_df[self.vs_df['date']==vdate].copy()
                tdmform_vdate = get_tdm_dateform(date_str=vdate)

                min_sbp = np.nanmin(vdf_frag['SBP (mmHg)'])
                max_sbp = np.nanmax(vdf_frag['SBP (mmHg)'])
                sbp_range_txt = f"{int(min_sbp) if not np.isnan(min_sbp) else ''}-{int(max_sbp) if not np.isnan(max_sbp) else ''}"

                min_dbp = np.nanmin(vdf_frag['DBP (mmHg)'])
                max_dbp = np.nanmax(vdf_frag['DBP (mmHg)'])
                dbp_range_txt = f"{int(min_dbp) if not np.isnan(min_dbp) else ''}-{int(max_dbp) if not np.isnan(max_dbp) else ''}"

                vs_text_dict["SBP/DBP"].append(f"({tdmform_vdate}){sbp_range_txt}/{dbp_range_txt}")

                min_hr = np.nanmin(vdf_frag['PR (회/min)'])
                max_hr = np.nanmax(vdf_frag['PR (회/min)'])
                hr_range_txt = f"{int(min_hr) if not np.isnan(min_hr) else ''}-{int(max_hr) if not np.isnan(max_hr) else ''}"

                vs_text_dict["HR"].append(f"({tdmform_vdate}){hr_range_txt}")

            vs_text = "\n".join([f"{k}: {' <-'.join(v)}" for k, v in vs_text_dict.items()])


        else:pass

        return vs_text


    def generate_tdm_reply_text(self):

        # file_name = f"{self.pt_dict['drug']}_{self.pt_dict['name']}_{self.pt_dict['id']}_{self.pt_dict['tdm_date'].replace('-','')}.txt"
        # file_path = f"{self.reply_text_saving_dir}/{file_name}"
        # file_content = ""

        basic_info_text = f"{self.pt_dict['id']} {self.pt_dict['name']} {self.pt_dict['sex']}/{self.pt_dict['age']} {self.pt_dict['height']}cm {self.pt_dict['weight']}kg {self.pt_dict['drug']}\n\n"
        # hx_text = f"*Hx.\n{self.pt_dict['history']}\n\n"
        hx_text = f"*Hx.\n\n"

        if self.pt_dict['drug']=='VCM':
            hd_text = f"*HD {self.pt_dict['hemodialysis']}\n\n"
            cslt_text = f"*IMI consult\n{self.pt_dict['consult']}\n\n"

            # self.pt_dict['vs'] = '(/) <-(/) <-(/) <-(/) <-(/)' # VS은 현재 테스트 중이라 임시로 입력
            self.pt_dict['vs'] = self.get_vs_text(drug=self.pt_dict['drug'])
            vs_text = f"{self.pt_dict['vs']}\n"

            lab_text = self.get_lab_text(drug=self.pt_dict['drug'])

            culture_date_test = get_tdm_dateform(date_str=self.tdm_date)
            culture_text = f"*Cx\nBlood C.: ({culture_date_test})(-)\nUrine: ({culture_date_test})(-)\nSputum: ({culture_date_test})(-)\n\n"

            drug_admin_text, prev_adm_date = self.get_drug_administration_hx_full_text(drug=self.pt_dict['drug'])
            drug_admin_text +="\n\n"

            base_lab_dict = self.get_base_cr_gfr_text(prev_adm_date=prev_adm_date)
            cr_base_tups = base_lab_dict['Cr']
            gfr_base_tups = base_lab_dict['GFR']

            base_lab_text = f"Cr(base): ({cr_base_tups[0]}){cr_base_tups[1]}\nGFR(base): ({gfr_base_tups[0]}){gfr_base_tups[1]}\n\n"

            conc_medi_txt = self.get_concomitant_mdx_text(drug=self.pt_dict['drug'])
            concomitant_medi_text = f"*CM\n{conc_medi_txt}\n\n" if conc_medi_txt!='' else f"*CM (-)\n\n"
            # concentration_text = f"*Concentration\n{'테스트중입니다'}\n\n"
            concentration_text = f"*Concentration\n{self.get_drug_concentration_text(drug=self.pt_dict['drug'])}\n\n"
            comment_text = f"*Comment\n{f'{self.tdm_writer} ({get_tdm_dateform(self.tdm_date)}) '}\n\n"

            # 알부민은 최근것만
            # 타과의뢰 의뢰문도 가져오기

            # GFR은 기록남길때는 MDRD로, 없으면 CKD-EPI로
            # Baseline : 투약직전에 Cr과 GFR (신환때 적고 이후에는 바뀌지 않는다!?) -> 투약력시간과 비교해서 기록해야할듯

            # Vanco : Cr 변화량이 얼마 기준 이상 (0.5?) 올라가면 ADR(adverse drug reaction) 신고해야함
            # 아미카신 : 이독성 병력있으면 신고해야함

            # 이전투약력 & 현투약력 : 최근 2주간의 투약력을 기록하되 시간순으로 '농도1 (datetime 1, datetime 2, datetime 3) 농도2 (datetime4, datetime5 ...) 최신것이 아래로 가도록 작성한다'
            # 채혈시간 : TDM리포트 or 검체텍스트(필터: 약물 및 중금속) 에도 나오는데, 가끔 둘이 다른 경우가 있어서 이런 경우에는 병동에 전화하여 채혈시간을 확인해야 한다.

            # 현재 투약 중단상태인지, 앞으로 투약이 어떻게 계획되어 있는지 정보가 오더에 없으면 확인해야 !

            etc_text = f"\nVd(L/kg) \nVc \nCL (ml/min/kg) \nCL(L/hr) \nt1/2(hr) \nVd ss \n\n==========================================================================\n= Drug concentration ( Target : {self.tdm_target_txt_dict[self.pt_dict['drug']]})\n1) 추정 Peak :  ㎍/mL\n2) 추정 Trough :  ㎍/mL\n3) 추정 AUC :  mg*h/L\n\n= Interpretation : \n\n\n= Recommendation : \n1. \n\n2. \n\n문의사항은 다음의 전화번호로.\n임상약리학과 (내선 T. 3956)/Pf. 정재용, 윤성혜 (내선 T. 3956)"


            self.file_content= basic_info_text+hx_text+hd_text+cslt_text+vs_text+base_lab_text+lab_text+culture_text+drug_admin_text+concomitant_medi_text + concentration_text + comment_text + etc_text

        elif self.pt_dict['drug'] == 'DGX':
            echo_text = f"*Echocardiography\n{self.get_echocardiography_text()}\n\n"
            ecg_text = f"*ECG\n{self.get_ecg_text()}\n\n"
            cslt_text = f"*IMC consult\n{self.pt_dict['consult']}\n\n"

            self.pt_dict['vs'] = self.get_vs_text(drug=self.pt_dict['drug'])
            vs_text = f"{self.pt_dict['vs']}\n"

            lab_text = self.get_lab_text(drug=self.pt_dict['drug'])

            # culture_text = f"*Cx\n{'테스트중입니다'}\n\n"

            # prev_admin_text = f"*이전 투약력\n{'테스트중입니다'}\n\n"
            # cur_admin_text = f"*현 투약력\n{'테스트중입니다'}\n\n"
            # drug_admin_text = prev_admin_text + cur_admin_text
            drug_admin_text, prev_adm_date = self.get_drug_administration_hx_full_text(drug=self.pt_dict['drug'])
            drug_admin_text += "\n\n"

            base_lab_dict = self.get_base_cr_gfr_text(prev_adm_date=prev_adm_date)
            cr_base_tups = base_lab_dict['Cr']
            gfr_base_tups = base_lab_dict['GFR']

            # base_lab_text = f"Cr(base): ({cr_base_tups[0]}){cr_base_tups[1]}\nGFR(base): ({gfr_base_tups[0]}){gfr_base_tups[1]}\n\n"

            conc_medi_txt = self.get_concomitant_mdx_text(drug=self.pt_dict['drug'])
            concomitant_medi_text = f"*CM\n{conc_medi_txt}\n\n" if conc_medi_txt!='' else f"*CM (-)\n\n"
            # concentration_text = f"*Concentration\n{'테스트중입니다'}\n\n"
            concentration_text = f"*Concentration\n{self.get_drug_concentration_text(drug=self.pt_dict['drug'])}\n\n"
            comment_text = f"*Comment\n{f'{self.tdm_writer} ({get_tdm_dateform(self.tdm_date)}) '}\n\n"

            dgx_caution_text = "* Digoxin의 혈중 약물농도만으로는 약효 및 독성 발현 산출에 한계가 있으므로, 임상증상을 뒷받침하는 참고자료로 활용하시기 바랍니다."
            etc_text = f"\nVd(L/kg) \nCL(L/hr) \nt1/2(hr) \n\n==========================================================================\n= Drug concentration ( Target : {self.tdm_target_txt_dict[self.pt_dict['drug']]})\n1) 추정 Peak :  ng/mL\n2) 추정 Trough :  ng/mL\n\n= Interpretation : \n\n\n= Recommendation : \n{dgx_caution_text}\n\n1. \n\n2. \n\n문의사항은 다음의 전화번호로.\n임상약리학과 (내선 T. 3956)/Pf. 정재용, 윤성혜 (내선 T. 3956)"

            self.file_content = basic_info_text + hx_text + cslt_text + echo_text + ecg_text + vs_text + lab_text + drug_admin_text + concomitant_medi_text + concentration_text + comment_text + etc_text

        elif self.pt_dict['drug'] in ('AMK', 'GTM'):
            hd_text = f"*HD {self.pt_dict['hemodialysis']}\n\n"
            cslt_text = f"*IMI consult\n{self.pt_dict['consult']}\n\n"

            self.pt_dict['vs'] = self.get_vs_text(drug=self.pt_dict['drug'])
            vs_text = f"{self.pt_dict['vs']}\n"

            lab_text = self.get_lab_text(drug=self.pt_dict['drug'])

            culture_date_test = get_tdm_dateform(date_str=self.tdm_date)
            culture_text = f"*Cx\nBlood C.: ({culture_date_test})(-)\nUrine: ({culture_date_test})(-)\nSputum: ({culture_date_test})(-)\n\n"

            drug_admin_text, prev_adm_date = self.get_drug_administration_hx_full_text(drug=self.pt_dict['drug'])
            drug_admin_text += "\n\n"

            base_lab_dict = self.get_base_cr_gfr_text(prev_adm_date=prev_adm_date)
            cr_base_tups = base_lab_dict['Cr']
            gfr_base_tups = base_lab_dict['GFR']

            base_lab_text = f"Cr(base): ({cr_base_tups[0]}){cr_base_tups[1]}\nGFR(base): ({gfr_base_tups[0]}){gfr_base_tups[1]}\n\n"

            conc_medi_txt = self.get_concomitant_mdx_text(drug=self.pt_dict['drug'])
            concomitant_medi_text = f"*CM\n{conc_medi_txt}\n\n" if conc_medi_txt!='' else f"*CM (-)\n\n"
            # concentration_text = f"*Concentration\n{'테스트중입니다'}\n\n"
            concentration_text = f"*Concentration\n{self.get_drug_concentration_text(drug=self.pt_dict['drug'])}\n\n"
            comment_text = f"*Comment\n{f'{self.tdm_writer} ({get_tdm_dateform(self.tdm_date)}) '}\n\n"

            etc_text = f"\nVd(L/kg) \nCL(L/hr) \nt1/2(hr) \n\n==========================================================================\n= Drug concentration ( Target : {self.tdm_target_txt_dict[self.pt_dict['drug']]})\n1) 추정 Peak :  ㎍/mL\n2) 추정 Trough :  ㎍/mL\n\n= Interpretation : \n\n\n= Recommendation : \n1. \n\n2. \n\n문의사항은 다음의 전화번호로.\n임상약리학과 (내선 T. 3956)/Pf. 정재용, 윤성혜 (내선 T. 3956)"

            self.file_content = basic_info_text + hx_text + hd_text + cslt_text + vs_text + base_lab_text + lab_text + culture_text + drug_admin_text + concomitant_medi_text + concentration_text + comment_text + etc_text

        elif self.pt_dict['drug'] == 'VPA':
            eeg_text = f"*EEG\n{self.get_eeg_text()}\n\n"
            cslt_text = f"*NR consult\n{self.pt_dict['consult']}\n\n"

            # self.pt_dict['vs'] = self.get_vs_text(drug=self.pt_dict['drug'])
            # vs_text = f"{self.pt_dict['vs']}\n"

            lab_text = self.get_lab_text(drug=self.pt_dict['drug'])

            # culture_text = f"*Cx\n{'테스트중입니다'}\n\n"

            # prev_admin_text = f"*이전 투약력\n{'테스트중입니다'}\n\n"
            # cur_admin_text = f"*현 투약력\n{'테스트중입니다'}\n\n"
            # drug_admin_text = prev_admin_text + cur_admin_text
            drug_admin_text, prev_adm_date = self.get_drug_administration_hx_full_text(drug=self.pt_dict['drug'])
            drug_admin_text += "\n\n"

            base_lab_dict = self.get_base_cr_gfr_text(prev_adm_date=prev_adm_date)
            cr_base_tups = base_lab_dict['Cr']
            gfr_base_tups = base_lab_dict['GFR']

            # base_lab_text = f"Cr(base): ({cr_base_tups[0]}){cr_base_tups[1]}\nGFR(base): ({gfr_base_tups[0]}){gfr_base_tups[1]}\n\n"

            conc_medi_txt = self.get_concomitant_mdx_text(drug=self.pt_dict['drug'])
            concomitant_medi_text = f"*CM\n{conc_medi_txt}\n\n" if conc_medi_txt!='' else f"*CM (-)\n\n"
            # concentration_text = f"*Concentration\n{'테스트중입니다'}\n\n"
            concentration_text = f"*Concentration\n{self.get_drug_concentration_text(drug=self.pt_dict['drug'])}\n\n"
            comment_text = f"*Comment\n{f'{self.tdm_writer} ({get_tdm_dateform(self.tdm_date)}) '}\n\n"

            dgx_caution_text = "* Digoxin의 혈중 약물농도만으로는 약효 및 독성 발현 산출에 한계가 있으므로, 임상증상을 뒷받침하는 참고자료로 활용하시기 바랍니다."
            etc_text = f"\nVd(L/kg) \nCL(L/hr) \nt1/2(hr) \n\n==========================================================================\n= Drug concentration ( Target : {self.tdm_target_txt_dict[self.pt_dict['drug']]})\n1) 추정 Peak :  ng/mL\n2) 추정 Trough :  ng/mL\n\n= Interpretation : \n\n\n= Recommendation : \n\n1. \n\n2. \n\n문의사항은 다음의 전화번호로.\n임상약리학과 (내선 T. 3956)/Pf. 정재용, 윤성혜 (내선 T. 3956)"

            self.file_content = basic_info_text + hx_text + cslt_text + eeg_text + lab_text + drug_admin_text + concomitant_medi_text + concentration_text + comment_text + etc_text


        else: pass
        # with open(file_path, "w", encoding="utf-8-sig") as f:
        #     f.write(self.file_content)

    def define_ir_info(self):

        self.wkday_dict = {0: '월', 1: '화', 2: '수', 3: '목', 4: '금', 5: '토', 6: '일'}

        self.ir_term_dict = {'VCM': {'half_life': 'Half-life (hr)',
                                     'vd_ss': 'Vd steady state (L)',
                                     'total_cl': 'Total Cl (L/hr)',
                                     'vc': 'Vc (L/kg)',
                                     'est_peak': '추정 peak 농도',
                                     'est_trough': '추정 trough 농도',
                                     'adm_amount': '1회 투약 용량(mg)',
                                     'adm_interval': '투약 간격(h)',
                                     },
                             'DGX': {'half_life': 'Half-life (hr)',
                                     'total_vd': 'Total Vd (L)',
                                     'total_cl': 'Total Cl (L/hr)',
                                     'est_peak': '추정 peak 농도',
                                     'est_trough': '추정 trough 농도',
                                     },
                             'AMK': {'half_life': 'Half-life (hr)',
                                     'total_vd': 'Total Vd (L)',
                                     'total_cl': 'Total Cl (L/hr)',
                                     'est_peak': '추정 peak 농도',
                                     'est_trough': '추정 trough 농도',
                                     },
                             'VPA': {'half_life': 'Half-life (hr)',
                                     'total_vd': 'Total Vd (L)',
                                     'total_cl': 'Total Cl (L/hr)',
                                     'est_peak': '추정 peak 농도',
                                     'est_trough': '추정 trough 농도',
                                     },
                             'GTM': {'half_life': 'Half-life (hr)',
                                     'total_vd': 'Total Vd (L)',
                                     'total_cl': 'Total Cl (L/hr)',
                                     'est_peak': '추정 peak 농도',
                                     'est_trough': '추정 trough 농도',
                                     },
                             }

        self.comed_recstr_dict = {'VCM': {'amikacin': 'vancomycin과 병용투여 시 이독성 위험이 증가하고 신기능을 저하시킬 수 있으므로, ',
                                          'gentamicin': 'vancomycin과 병용투여 시 이독성 위험이 증가하고 신기능을 저하시킬 수 있으므로, ',
                                          'amphotericin': 'vancomycin과 병용투여 시 신기능을 저하시킬 수 있으므로, ',
                                          'cyclosporine': 'vancomycin과 병용투여 시 신기능을 저하시킬 수 있으므로, ',
                                          'tacrolimus': 'vancomycin과 병용투여 시 신기능을 저하시킬 수 있으므로, ',
                                          'piperacillin/tazobactam': 'vancomycin과 병용투여 시 신기능을 저하시킬 수 있으므로, ',
                                          'meropenem': 'vancomycin과 병용투여 시 신기능을 저하시킬 수 있으므로, ',

                                          'CRRT': '상환 지속적 신대체요법 적용중으로, CRRT setting 변화에 따른 약물농도의 변동폭이 크므로, ',
                                          '영아': '영아의 경우 성장에 따른 약동학 파라미터 변화의 폭이 크므로, ',
                                          '유아': '유아의 경우 성장에 따른 약동학 파라미터 변화의 폭이 크므로, ',
                                          '소아': '소아의 경우 성장에 따른 약동학 파라미터 변화의 폭이 크므로, ',
                                          },
                                  'AMK': {'penicillin': 'amikacin과 병용투여 시 이독성 위험이 증가하고 신기능을 저하시킬 수 있으므로, ',
                                          'ampicillin': 'amikacin과 병용투여 시 이독성 위험이 증가하고 신기능을 저하시킬 수 있으므로, ',
                                          'nafcillin': 'amikacin과 병용투여 시 신기능을 저하시킬 수 있으므로, ',
                                          'oxacillin': 'amikacin과 병용투여 시 신기능을 저하시킬 수 있으므로, ',
                                          'carbenecillin': 'amikacin과 병용투여 시 신기능을 저하시킬 수 있으므로, ',

                                          'CRRT': '상환 지속적 신대체요법 적용중으로, CRRT setting 변화에 따른 약물농도의 변동폭이 크므로, ',
                                          '영아': '영아의 경우 성장에 따른 약동학 파라미터 변화의 폭이 크므로, ',
                                          '유아': '유아의 경우 성장에 따른 약동학 파라미터 변화의 폭이 크므로, ',
                                          '소아': '소아의 경우 성장에 따른 약동학 파라미터 변화의 폭이 크므로, ',
                                          },
                                  'GTM': {'penicillin': 'gentamicin과 병용투여 시 synergistic effect가 발생할 수 있으므로, ',
                                          'ampicillin': 'gentamicin과 병용투여 시 synergistic effect가 발생할 수 있으므로, ',
                                          'nafcillin': 'gentamicin과 병용투여 시 synergistic effect가 발생할 수 있으므로, ',
                                          'oxacillin': 'gentamicin과 병용투여 시 synergistic effect가 발생할 수 있으므로, ',
                                          'carbenecillin': 'gentamicin과 병용투여 시 synergistic effect가 발생할 수 있으므로, ',

                                          'CRRT': '상환 지속적 신대체요법 적용중으로, CRRT setting 변화에 따른 약물농도의 변동폭이 크므로, ',
                                          '영아': '영아의 경우 성장에 따른 약동학 파라미터 변화의 폭이 크므로, ',
                                          '유아': '유아의 경우 성장에 따른 약동학 파라미터 변화의 폭이 크므로, ',
                                          '소아': '소아의 경우 성장에 따른 약동학 파라미터 변화의 폭이 크므로, ',
                                          },
                                  'DGX': {},
                                  'VPA': {},
                                  'PHT': {},
                                  }

        # mode = 'manual'
        # drug = 'VCM'

        # ss = 'SS'
        # conclusion='Therapeutic'
        # method = '용법변경'

        self.interpretation_dict = {'SS': {'Subtherapeutic': 'Steady state, Subtherapeutic Level',
                                           'Lowermargin': 'Steady state, Lower margin of Therapeutic Level',
                                           'LMofPeak': 'Steady state, Lower margin of Therapeutic Peak Level',
                                           'Therapeutic': 'Steady state, Therapeutic Level',
                                           'Uppermargin': 'Steady state, Upper margin of Therapeutic Level',
                                           'UMofPeak': 'Steady state, Upper margin of Therapeutic Peak Level',
                                           'Toxic': 'Steady state, Toxic Level',
                                           'ToxicPeak': 'Steady state, Toxic Peak Level',
                                           'Holding': '투약 중단 상태',
                                           },
                                    'NSS': {'Subtherapeutic': 'Non-steady state, Subtherapeutic Level expected',
                                            'Lowermargin': 'Non-steady state, Lower margin of Therapeutic Level expected',
                                            'LMofPeak': 'Non-steady state, Lower margin of Therapeutic Peak Level expected',
                                            'Therapeutic': 'Non-steady state, Therapeutic Level expected',
                                            'Uppermargin': 'Non-steady state, Upper margin of Therapeutic Level expected',
                                            'UMofPeak': 'Non-steady state, Upper margin of Therapeutic Peak Level expected',
                                            # 'Toxic': 'Non-steady state, Toxic Level expected',
                                            'Toxic': 'Non-steady state, Toxic Level',
                                            'ToxicPeak': 'Non-steady state, Toxic Peak Level',
                                            'Holding': '투약 중단 상태'
                                            },
                                    }

        self.ir_recomm_dict = {'AMK': {'Subtherapeutic': {'rec1_SS': '1. Subtherapeutic level입니다. 용법 변경 권장합니다.\n\n',
                                                          'rec1_NSS': '1. 아직 항정상태에 도달하지 않았으나 현 용법 유지시 Subtherapeutic level에 이를 것으로 예상됩니다. 용법 변경 권장합니다.\n\n',
                                                          'rec2_공통': f'2. 적극적인 감염 조절을 위하여 다음 투약부터 [변경약물용법(용량,투약방식,투여간격)] 로 용법 변경 권장하며, 이 때 예상되는 항정상태에서의 농도는 아래와 같습니다.\n\n3. 신기능 및 임상상의 변화에 유의하시고, [f/u날짜(요일)] 투약 전 30분 및 투약 후 30분에 각각 1회씩 채혈을 통해 TDM f/u하시어 변경 용법의 적절성을 재확인하시기 바랍니다.',
                                                          },
                                       'Lowermargin': {
                                           'rec1_SS': '1. Lower margin of therapeutic Level 입니다. 감염 조절이 원활하다면 당분간 현 용법 유지 가능합니다.\n\n',
                                           'rec1_NSS': '1. 아직 항정상태에 도달하지 않았으나 현 용법 유지시 Lower margin of therapeutic level에 이를 것으로 예상됩니다. 감염 조절이 원활하다면 당분간 현 용법 유지 가능합니다.\n\n',
                                           'rec2_공통': '2. 적극적인 감염 조절을 위하여 용법 변경을 고려하시는 경우에는 다음 투약부터 [변경약물용법(용량,투약방식,투여간격)] 로 용법 변경 가능하며, 이 때 예상되는 항정상태에서의 농도는 아래와 같습니다.\n\n3. 신기능 및 임상상의 변화에 유의하시고, [f/u날짜(요일)] 투약 전 30분 및 투약 후 30분에 각각 1회씩 채혈을 통해 TDM f/u하시어 용법의 적절성을 재확인하시기 바랍니다.',
                                           },
                                       'LMofPeak': {
                                           'rec1_SS': '1. Lower margin of therapeutic peak level 입니다. 감염 조절이 원활하다면 당분간 현 용법 유지 가능합니다.\n\n',
                                           'rec1_NSS': '1. 아직 항정상태에 도달하지 않았으나 현 용법 유지시 Lower margin of therapeutic peak level에 이를 것으로 예상됩니다. 감염 조절이 원활하다면 당분간 현 용법 유지 가능합니다.\n\n',
                                           'rec2_공통': '2. 적극적인 감염 조절을 위하여 용법 변경을 고려하시는 경우에는 다음 투약부터 [변경약물용법(용량,투약방식,투여간격)] 로 용법 변경 가능하며, 이 때 예상되는 항정상태에서의 농도는 아래와 같습니다.\n\n3. 신기능 및 임상상의 변화에 유의하시고, [f/u날짜(요일)] 투약 전 30분 및 투약 후 30분에 각각 1회씩 채혈을 통해 TDM f/u하시어 용법의 적절성을 재확인하시기 바랍니다.',
                                           },
                                       'Therapeutic': {'rec1_SS': '1. Therapeutic level 입니다. 현 용법 유지 가능합니다.\n\n',
                                                       'rec1_NSS': '1. 아직 항정상태에 도달하지 않았으나 현 용법 유지시 Therapeutic level에 이를 것으로 예상됩니다. 현 용법 유지 가능합니다.\n\n',
                                                       'rec2_공통': '2. 신기능 및 임상상의 변화에 유의하시고, [f/u날짜(요일)] 투약 전 30분 및 투약 후 30분에 각각 1회씩 채혈을 통해 TDM f/u하시어 유지 용법의 적절성을 재확인하시기 바랍니다.',
                                                       },
                                       'Uppermargin': {
                                           'rec1_SS': '1. Upper margin of therapeutic level 입니다. 적극적인 감염 조절이 필요한 경우, 당분간 현 용법 유지 가능합니다.\n\n',
                                           'rec1_NSS': '1. 아직 항정상태에 도달하지 않았으나 현 용법 유지시 Upper margin of therapeutic level에 이를 것으로 예상되나 적극적인 증상 조절이 필요한 경우, 당분간 현 용법 유지 가능합니다.\n\n',
                                           'rec2_공통': '2. 독성증상 발현 예방을 위하여 용법 변경을 고려하시는 경우에는 다음 투약부터 [변경약물용법(용량,투약방식,투여간격)] 로 용법 변경 가능하며, 이 때 예상되는 항정상태에서의 농도는 아래와 같습니다.\n\n3. 신기능 및 임상상의 변화에 유의하시고, [f/u날짜(요일)] 투약 전 30분 및 투약 후 30분에 각각 1회씩 채혈을 통해 TDM f/u하시어 용법의 적절성을 재확인하시기 바랍니다.',
                                           },
                                       'UMofPeak': {
                                           'rec1_SS': '1. Upper margin of therapeutic peak level 입니다. 적극적인 감염 조절이 필요한 경우, 당분간 현 용법 유지 가능합니다.\n\n',
                                           'rec1_NSS': '1. 아직 항정상태에 도달하지 않았으나 현 용법 유지시 Upper margin of therapeutic peak level에 이를 것으로 예상되나 적극적인 증상 조절이 필요한 경우, 당분간 현 용법 유지 가능합니다.\n\n',
                                           'rec2_공통': '2. 독성증상 발현 예방을 위하여 용법 변경을 고려하시는 경우에는 다음 투약부터 [변경약물용법(용량,투약방식,투여간격)] 로 용법 변경 가능하며, 이 때 예상되는 항정상태에서의 농도는 아래와 같습니다.\n\n3. 신기능 및 임상상의 변화에 유의하시고, [f/u날짜(요일)] 투약 전 30분 및 투약 후 30분에 각각 1회씩 채혈을 통해 TDM f/u하시어 용법의 적절성을 재확인하시기 바랍니다.',
                                           },
                                       'Toxic': {'rec1_SS': '1. Toxic level입니다. ',
                                                 'rec1_NSS-nontoxic': '1. 아직 항정상태에 도달하지 않았으나 현 용법 유지시 Toxic level에 이를 것으로 예상됩니다. ',
                                                 'rec1_NSS-toxic': '1. 아직 항정상태에 도달하지 않았으나 현재 Toxic level입니다. ',
                                                 'rec2_용법변경': '용법 변경 권장합니다.\n\n2. 독성증상 발현 예방을 위하여 다음 투약부터 [변경약물용법(용량,투약방식,투여간격)] 로 용법 변경 가능하며, 이 때 예상되는 항정상태에서의 농도는 아래와 같습니다.\n\n3. 신기능 및 임상상의 변화에 유의하시고, [f/u날짜(요일)] 투약 전 30분 및 투약 후 30분에 각각 1회씩 채혈을 통해 TDM f/u하시어 용법 변경의 적절성을 재확인하시기 바랍니다.',
                                                 'rec2_휴약후변경': '휴약 및 용법 변경 권장합니다.\n\n2. 독성증상 발현 예방을 위하여 예정되어있는 다음 투약 [기존예정된투약일(요일,시간)] 부터 휴약하시고 약물 농도가 충분히 감소할 것으로 예상되는 [변경용법적용날짜(요일,시간,예상농도)] 이후부터 [변경약물용법(용량,투약방식,투여간격)] 로 용법 변경 가능하며, 이 때 예상되는 항정상태에서의 농도는 아래와 같습니다.\n\n3. 신기능 및 임상상의 변화에 유의하시고, [f/u날짜(요일)] 투약 전 30분 및 투약 후 30분에 각각 1회씩 채혈을 통해 TDM f/u하시어 용법 변경의 적절성을 재확인하시기 바랍니다.',
                                                 'rec2_휴약후f/u': '휴약 권장합니다.\n\n2. 신기능 및 임상상의 변화에 유의하시고, 휴약을 유지하시다가, [f/u날짜(요일)] 정규 채혈을 통한 TDM f/u하시어 투약 재개 시점 및 재개 용법을 재확인하시기 바랍니다.',
                                                 },
                                       'Holding': {'rec1_SS': '1. 상환 [마지막투약일(요일, 시간)] 마지막 투약 이후 투약 중단 상태이며, 측정된 약물 농도는 [측정시 약물 농도 (Toxic level/Non-toxic level)] 입니다. ',
                                                   'rec1_NSS': '1. 상환 [마지막투약일(요일, 시간)] 마지막 투약 이후 투약 중단 상태이며, 측정된 약물 농도는 [측정시 약물 농도 (Toxic level/Non-toxic level)] 입니다. \n\n',
                                                   'rec2_Toxic정규채혈f/u': '휴약 유지 권장합니다.\n\n2. 신기능 및 임상상의 변화에 주의하시고, [f/u날짜(요일)] 정규 채혈을 통한 TDM f/u하시어 용법의 적절성을 재평가하시기 바랍니다',
                                                   'rec2_Toxic곧Non-toxic투약재개': '그러나 이를 기반으로 산출되는 [당일 예상 투약시점 (시간)] 농도는 [당일 예상 투약시점의 예상 농도] mcg/mL 로 Non-toxic level 예상되어 이후 투약 재개해볼 수 있습니다.\n\n2. 투약 재개를 고려하시는 경우 휴약을 유지하시다가 [변경용법적용날짜(요일,시간,예상농도)] 이후부터 [변경약물용법(용량,투약방식,투여간격)] 로 재개 가능합니다. 현재 약물배설능이 유지됨을 가정할 때 예상되는 항정상태 농도는 아래와 같습니다.\n\n3. 신기능 및 임상상의 변화에 유의하시고, [f/u날짜(요일)] 투약 전 30분 및 투약 후 30분에 각각 1회씩 채혈을 통해 TDM f/u하시어 용법의 적절성을 재확인하시기 바랍니다.',
                                                   'rec2_NonToxic투약재개': '투약 재개 가능합니다.\n\n2. 투약 재개를 고려하시는 경우 [변경용법적용날짜(요일,시간,예상농도)] 이후부터 [변경약물용법(용량,투약방식,투여간격)] 로 재개 가능합니다. 현재 약물배설능이 유지됨을 가정할 때 예상되는 항정상태 농도는 아래와 같습니다.\n\n3. 신기능 및 임상상의 변화에 유의하시고, [f/u날짜(요일)] 투약 전 30분 및 투약 후 30분에 각각 1회씩 채혈을 통해 TDM f/u하시어 용법의 적절성을 재확인하시기 바랍니다.',
                                           },
                                       # 'Below The Detection Limit': {
                                       #     'rec1_SS': '1. 오늘 측정된 약물 농도는 Below the detection limit 으로 약동학적 파라미터 산출이 불가능합니다.\n\n',
                                       #     'rec1_NSS': '1. 오늘 측정된 약물 농도는 Below the detection limit 으로 약동학적 파라미터 산출이 불가능합니다.\n\n',
                                       #     'rec2_정규채혈f/u': '2. 투약 재개를 고려하시는 경우 신기능 및 임상상의 변화에 주의하시고, [f/u날짜(요일)] 정규 채혈을 통한 TDM f/u하시어 용법의 적절성을 재평가하시기 바랍니다',
                                       #     'rec2_투약중단상태(현재)': '2. 투약 재개를 고려하시는 경우 단계적 증량 위해 [변경약물용법(용량,투약방식,투여간격)] 로 시작해볼 수 있습니다.\n\n3. 신기능 및 임상상의 변화에 유의하시고, [f/u날짜(요일)] 투약 전 30분 및 투약 후 30분에 각각 1회씩 채혈을 통해 TDM f/u하시어 용법의 적절성을 재확인하시기 바랍니다.',
                                       #     'rec2_용법추천(subthera명확시)': '2. 규칙적인 투약력 고려하였을 때 적극적인 치료를 위하여 용법 변경을 고려하시는 경우에는 다음 예정된 투약부터 [변경약물용법(용량,투약방식,투여간격)] 로 변경해볼 수 있습니다.\n\n3. 신기능 및 임상상의 변화에 유의하시고, [f/u날짜(요일)] 투약 전 30분 및 투약 후 30분에 각각 1회씩 채혈을 통해 TDM f/u하시어 용법의 적절성을 재확인하시기 바랍니다.',
                                       # 
                                       #     },
                                       },
                               'VCM': {'Subtherapeutic': {'rec1_SS': '1. Subtherapeutic level입니다. 용법 변경 권장합니다.\n\n',
                                                          'rec1_NSS': '1. 아직 항정상태에 도달하지 않았으나 현 용법 유지시 Subtherapeutic level에 이를 것으로 예상됩니다. 용법 변경 권장합니다.\n\n',
                                                          'rec2_공통': '2. 적극적인 감염 조절을 위하여 다음 투약부터 [변경약물용법(용량,투약방식,투여간격)] 로 용법 변경 권장하며, 이 때 예상되는 항정상태에서의 농도는 아래와 같습니다.\n\n3. 신기능 및 임상상의 변화에 유의하시고, [f/u날짜(요일)] 투약 30분전 및 투약 1시간 후 채혈로 TDM f/u하시어 변경 용법의 적절성을 재확인하시기 바랍니다.',
                                                          },
                                       'Lowermargin': {
                                           'rec1_SS': '1. Lower margin of therapeutic Level 입니다. 감염 조절이 원활하다면 당분간 현 용법 유지 가능합니다.\n\n',
                                           'rec1_NSS': '1. 아직 항정상태에 도달하지 않았으나 현 용법 유지시 Lower margin of therapeutic level에 이를 것으로 예상됩니다. 감염 조절이 원활하다면 당분간 현 용법 유지 가능합니다.\n\n',
                                           'rec2_공통': '2. 적극적인 감염 조절을 위하여 용법 변경을 고려하시는 경우에는 다음 투약부터 [변경약물용법(용량,투약방식,투여간격)] 로 용법 변경 가능하며, 이 때 예상되는 항정상태에서의 농도는 아래와 같습니다.\n\n3. 신기능 및 임상상의 변화에 유의하시고, [f/u날짜(요일)] 투약 30분전 및 투약 1시간 후 채혈로 TDM f/u하시어 용법의 적절성을 재확인하시기 바랍니다.',
                                           },
                                       'Therapeutic': {'rec1_SS': '1. Therapeutic level 입니다. 현 용법 유지 가능합니다.\n\n',
                                                       'rec1_NSS': '1. 아직 항정상태에 도달하지 않았으나 현 용법 유지시 Therapeutic level에 이를 것으로 예상됩니다. 현 용법 유지 가능합니다.\n\n',
                                                       'rec2_공통': '2. 신기능 및 임상상의 변화에 유의하시고, [f/u날짜(요일)] 투약 30분전 및 투약 1시간 후 채혈로 TDM f/u하시어 유지 용법의 적절성을 재확인하시기 바랍니다.',
                                                       },

                                       'Highpeak': {
                                           'rec1_SS': '1. Therapeutic level이나, 다소 높은 peak level로 인한 독성 증상의 예방을 위하여 용법 변경 권장합니다.\n\n',
                                           'rec1_NSS': '1. Therapeutic level에 이를 것으로 예상되나, 다소 높은 peak level로 인한 독성 증상의 예방을 위하여 용법 변경 권장합니다.\n\n',
                                           'rec2_공통': '2. 독성 증상 발현 예방을 위하여 다음 투약부터 [변경약물용법(용량,투약방식,투여간격)] 로 용법 변경 가능하며, 이 때 예상되는 항정상태에서의 농도는 아래와 같습니다.\n\n3. 신기능 및 임상상의 변화에 유의하시고, [f/u날짜(요일)] 투약 30분전 및 투약 1시간 후 채혈로 TDM f/u하시어 용법의 적절성을 재확인하시기 바랍니다.',
                                           },
                                       'Uppermargin': {
                                           'rec1_SS': '1. Upper margin of therapeutic level 입니다. 적극적인 감염 조절이 필요한 경우, 당분간 현 용법 유지 가능합니다.\n\n',
                                           'rec1_NSS': '1. 아직 항정상태에 도달하지 않았으나 현 용법 유지시 Upper margin of therapeutic level에 이를 것으로 예상되나 적극적인 증상 조절이 필요한 경우, 당분간 현 용법 유지 가능합니다.\n\n',
                                           'rec2_공통': '2. 독성증상 발현 예방을 위하여 용법 변경을 고려하시는 경우에는 다음 투약부터 [변경약물용법(용량,투약방식,투여간격)] 로 용법 변경 가능하며, 이 때 예상되는 항정상태에서의 농도는 아래와 같습니다.\n\n3. 신기능 및 임상상의 변화에 유의하시고, [f/u날짜(요일)] 투약 30분전 및 투약 1시간 후 채혈로 TDM f/u하시어 용법의 적절성을 재확인하시기 바랍니다.', },
                                       'Toxic': {'rec1_SS': '1. Toxic level입니다. ',
                                                 'rec1_NSS-nontoxic': '1. 아직 항정상태에 도달하지 않았으나 현 용법 유지시 Toxic level에 이를 것으로 예상됩니다. ',
                                                 'rec1_NSS-toxic': '1. 아직 항정상태에 도달하지 않았으나 현재 Toxic level입니다. ',
                                                 'rec2_용법변경': '용법 변경 권장합니다.\n\n2. 독성증상 발현 예방을 위하여 다음 투약부터 [변경약물용법(용량,투약방식,투여간격)] 로 용법 변경 가능하며, 이 때 예상되는 항정상태에서의 농도는 아래와 같습니다.\n\n3. 신기능 및 임상상의 변화에 유의하시고, [f/u날짜(요일)] 투약 30분전 및 투약 1시간 후 채혈로 TDM f/u하시어 용법 변경의 적절성을 재확인하시기 바랍니다.',
                                                 'rec2_휴약후변경': '휴약 및 용법 변경 권장합니다.\n\n2. 독성증상 발현 예방을 위하여 예정되어있는 다음 투약 [기존예정된투약일(요일,시간)] 부터 휴약하시고 약물 농도가 충분히 감소할 것으로 예상되는 [변경용법적용날짜(요일,시간,예상농도)] 이후부터 [변경약물용법(용량,투약방식,투여간격)] 로 용법 변경 가능하며, 이 때 예상되는 항정상태에서의 농도는 아래와 같습니다.\n\n3. 신기능 및 임상상의 변화에 유의하시고, [f/u날짜(요일)] 투약 30분전 및 투약 1시간 후 채혈로 TDM f/u하시어 용법 변경의 적절성을 재확인하시기 바랍니다.',
                                                 'rec2_휴약후f/u': '휴약 권장합니다.\n\n2. 신기능 및 임상상의 변화에 유의하시고, 휴약을 유지하시다가, [f/u날짜(요일)] 정규 채혈을 통한 TDM f/u하시어 투약 재개 시점 및 재개 용법을 재확인하시기 바랍니다.',
                                                 },
                                       'Holding': {'rec1_SS': '1. 상환 [마지막투약일(요일, 시간)] 마지막 투약 이후 투약 중단 상태이며, 측정된 약물 농도는 [측정시 약물 농도 (Toxic level/Non-toxic level)] 입니다. ',
                                                   'rec1_NSS': '1. 상환 [마지막투약일(요일, 시간)] 마지막 투약 이후 투약 중단 상태이며, 측정된 약물 농도는 [측정시 약물 농도 (Toxic level/Non-toxic level)] 입니다. \n\n',
                                                   'rec2_Toxic정규채혈f/u': '휴약 유지 권장합니다.\n\n2. 신기능 및 임상상의 변화에 주의하시고, [f/u날짜(요일)] 정규 채혈을 통한 TDM f/u하시어 용법의 적절성을 재평가하시기 바랍니다',
                                                   'rec2_Toxic곧Non-toxic투약재개': '그러나 이를 기반으로 산출되는 [당일 예상 투약시점 (시간)] 농도는 [당일 예상 투약시점의 예상 농도] mcg/mL 로 Non-toxic level 예상되어 이후 투약 재개해볼 수 있습니다.\n\n2. 투약 재개를 고려하시는 경우 휴약을 유지하시다가 [변경용법적용날짜(요일,시간,예상농도)] 이후부터 [변경약물용법(용량,투약방식,투여간격)] 로 재개 가능합니다. 현재 약물배설능이 유지됨을 가정할 때 예상되는 항정상태 농도는 아래와 같습니다.\n\n3. 신기능 및 임상상의 변화에 유의하시고, [f/u날짜(요일)] 투약 전 30분 및 투약 1 시간 후 채혈로 TDM f/u하시어 용법의 적절성을 재확인하시기 바랍니다.',
                                                   'rec2_NonToxic투약재개': '투약 재개 가능합니다.\n\n2. 투약 재개를 고려하시는 경우 [변경용법적용날짜(요일,시간,예상농도)] 이후부터 [변경약물용법(용량,투약방식,투여간격)] 로 재개 가능합니다. 현재 약물배설능이 유지됨을 가정할 때 예상되는 항정상태 농도는 아래와 같습니다.\n\n3. 신기능 및 임상상의 변화에 유의하시고, [f/u날짜(요일)] 투약 전 30분 및 투약 1시간 후 채혈로 TDM f/u하시어 용법의 적절성을 재확인하시기 바랍니다.',
                                           },
                                       },
                               'DGX': {'Subtherapeutic': {'rec1_SS': '1. Subtherapeutic level입니다. 용법 변경 권장합니다.\n\n',
                                                          'rec1_NSS': '1. 아직 항정상태에 도달하지 않아 정확한 약동학적 파라미터의 산출에 제한이 있으나, 현 용법 유지 시 Subtherapeutic level에 이를 것으로 예상됩니다. 용법 변경 권장합니다.\n\n',
                                                          'rec2_공통': '2. 적극적인 증상 조절을 위하여 다음 투약부터 [변경약물용법(용량,투약방식,투여간격)] 로 용법 변경 권장하며, 이 때 예상되는 항정상태에서의 농도는 아래와 같습니다.\n\n3. K/Ca level, 신기능 및 임상상의 변화에 유의하시고, [f/u날짜(요일)] 투약 30분 전 채혈을 통한 TDM f/u하시어 용법의 적절성을 재확인하시기 바랍니다. 약물의 효능이 기대 이하이거나 독성 의심 증상 발현 시 f/u 시기를 앞당길 수 있습니다. ',
                                                          },
                                       'Lowermargin': {
                                           'rec1_SS': '1. Lower margin of therapeutic level입니다. 증상 조절이 원활하다면 현 용법을 유지할 수 있습니다.\n\n',
                                           'rec1_NSS': '1. 아직 항정상태에 도달하지 않아 정확한 약동학적 파라미터의 산출에 제한이 있으나, 현 용법 유지 시 Lower margin of therapeutic level에 이를 것으로 예상됩니다. 증상 조절이 원활하다면 당분간 현 용법 유지 가능합니다.\n\n',
                                           'rec2_공통': '2. 적극적인 증상 조절을 위해 용법 변경을 고려하시는 경우에는 다음 투약부터 [변경약물용법(용량,투약방식,투여간격)] 로 용법 변경 가능하며, 이 때 예상되는 항정상태에서의 농도는 아래와 같습니다.\n\n3. K/Ca level, 신기능 및 임상상의 변화에 유의하시고, [f/u날짜(요일)] 투약 30분 전 채혈을 통한 TDM f/u하시어 용법의 적절성을 재확인하시기 바랍니다. 약물의 효능이 기대 이하이거나 독성 의심 증상 발현 시 f/u 시기를 앞당길 수 있습니다. ',
                                           },
                                       'Therapeutic': {'rec1_SS': '1. Therapeutic level 입니다. 현 용법 유지 가능합니다.\n\n',
                                                       'rec1_NSS': '1. 아직 항정상태에 도달하지 않아 정확한 약동학적 파라미터의 산출에 제한이 있으나, 현 용법 유지 시 Therapeutic level에 이를 것으로 예상됩니다. 현 용법 유지 가능합니다.\n\n',
                                                       'rec2_공통': '2. K/Ca level, 신기능 및 임상상의 변화에 유의하시고, [f/u날짜(요일)] 투약 30분 전 채혈을 통한 TDM f/u하시어 용법의 적절성을 재확인하시기 바랍니다. 약물의 효능이 기대 이하이거나 독성 의심 증상 발현 시 f/u 시기를 앞당길 수 있습니다. ',
                                                       },
                                       'Uppermargin': {
                                           'rec1_SS': '1. Upper margin of therapeutic level 입니다. 적극적인 증상 조절이 필요한 경우, 당분간 현 용법 유지 가능합니다.\n\n',
                                           'rec1_NSS': '1. 아직 항정상태에 도달하지 않아 정확한 약동학적 파라미터의 산출에 제한이 있으나, 현 용법 유지 시 Upper margin of therapeutic level에 이를 것으로 예상됩니다. 적극적인 증상 조절이 필요한 경우, 당분간 현 용법 유지 가능합니다.\n\n',
                                           'rec2_공통': '2. 독성증상 발현 예방을 위하여 용법 변경을 고려하시는 경우에는 다음 투약부터 [변경약물용법(용량,투약방식,투여간격)] 로 용법 변경 가능하며, 이 때 예상되는 항정상태에서의 농도는 아래와 같습니다.\n\n3. K/Ca level, 신기능 및 임상상의 변화에 유의하시고, [f/u날짜(요일)] 투약 30분 전 채혈을 통한 TDM f/u하시어 용법의 적절성을 재확인하시기 바랍니다. 약물의 효능이 기대 이하이거나 독성 의심 증상 발현 시 f/u 시기를 앞당길 수 있습니다. ', },
                                       'Toxic': {'rec1_SS': '1. Toxic level입니다. ',
                                                 'rec1_NSS-nontoxic': '1. 아직 항정상태에 도달하지 않아 정확한 약동학적 파라미터의 산출에 제한이 있으나, 현 용법 유지 시 Toxic level에 도달할 것으로 예상됩니다. ',
                                                 'rec1_NSS-toxic': '1. 아직 항정상태에 도달하지 않아 정확한 약동학적 파라미터의 산출에 제한이 있으나, 현재 Toxic level입니다. ',
                                                 'rec2_용법변경': '용법 변경 권장합니다.\n\n2. 독성증상 발현 예방을 위하여 다음 투약 [기존예정된투약일(요일,시간)] 부터 [변경약물용법(용량,투약방식,투여간격)] 로 용법 변경 가능하며, 이 때 예상되는 항정상태에서의 농도는 아래와 같습니다.\n\n3. K/Ca level, 신기능 및 임상상의 변화에 유의하시고, [f/u날짜(요일)] 투약 30분 전 채혈을 통한 TDM f/u하시어 용법의 적절성을 재확인하시기 바랍니다. 약물의 효능이 기대 이하이거나 독성 의심 증상 발현 시 f/u 시기를 앞당길 수 있습니다. ',
                                                 'rec2_휴약후변경': '휴약 및 용법 변경 권장합니다.\n\n2. 독성증상 발현 예방을 위하여 예정되어있는 다음 투약 [기존예정된투약일(요일,시간)] 부터 휴약하시고 약물 농도가 충분히 감소할 것으로 예상되는 [변경용법적용날짜(요일,시간,예상농도)] 이후부터 [변경약물용법(용량,투약방식,투여간격)] 로 용법 변경 가능하며, 이 때 예상되는 항정상태에서의 농도는 아래와 같습니다.\n\n3. K/Ca level, 신기능 및 임상상의 변화에 유의하시고, [f/u날짜(요일)] 투약 30분 전 채혈을 통한 TDM f/u하시어 용법의 적절성을 재확인하시기 바랍니다. 약물의 효능이 기대 이하이거나 독성 의심 증상 발현 시 f/u 시기를 앞당길 수 있습니다. ',
                                                 'rec2_휴약후f/u': '휴약 권장합니다.\n\n2. K/Ca level, 신기능 및 임상상의 변화에 유의하시고, 휴약을 유지하시다가, [f/u날짜(요일)] 정규채혈로 TDM 재의뢰하시어 재개 가능 여부를 재확인 하시기 바랍니다.',
                                                 },
                                       },
                               'VPA': {'Subtherapeutic': {'rec1_SS': '1. Subtherapeutic level입니다. 용법 변경 권장합니다.\n\n',
                                                          'rec1_NSS': '1. 아직 항정상태에 도달하지 않아 정확한 약동학적 파라미터의 산출에 제한이 있으나, 현 용법 유지 시 Subtherapeutic level에 이를 것으로 예상됩니다. 용법 변경 권장합니다.\n\n',
                                                          'rec2_공통': '2. 적극적인 증상 조절을 위하여 다음 투약부터 [변경약물용법(용량,투약방식,투여간격)]로 용법 변경 권장하며, 이 때 예상되는 항정상태에서의 농도는 아래와 같습니다.\n\n3. 암모니아 농도를 포함한 간기능 및 임상상의 변화에 유의하시고, [f/u날짜(요일)] 투약 30분 전 채혈을 통한 TDM f/u하시어 용법의 적절성을 재확인하시기 바랍니다. 약물의 효능이 기대 이하이거나 독성 의심 증상 발현 시 f/u 시기를 앞당길 수 있습니다. ',
                                                          },
                                       'Lowermargin': {
                                           'rec1_SS': '1. Lower margin of therapeutic level입니다. 증상 조절이 원활하다면 현 용법을 유지할 수 있습니다.\n\n',
                                           'rec1_NSS': '1. 아직 항정상태에 도달하지 않아 정확한 약동학적 파라미터의 산출에 제한이 있으나, 현 용법 유지 시 Lower margin of therapeutic level에 이를 것으로 예상됩니다. 증상 조절이 원활하다면 당분간 현 용법 유지 가능합니다.\n\n',
                                           'rec2_공통': '2. 적극적인 증상 조절을 위해 용법 변경을 고려하시는 경우에는 다음 투약부터 [변경약물용법(용량,투약방식,투여간격)] 로 용법 변경 가능하며, 이 때 예상되는 항정상태에서의 농도는 아래와 같습니다.\n\n3. 암모니아 농도를 포함한 간기능 및 임상상의 변화에 유의하시고, [f/u날짜(요일)] 투약 30분 전 채혈을 통한 TDM f/u하시어 용법의 적절성을 재확인하시기 바랍니다.',
                                           },
                                       'Therapeutic': {'rec1_SS': '1. Therapeutic level 입니다. 현 용법 유지 가능합니다.\n\n',
                                                       'rec1_NSS': '1. 아직 항정상태에 도달하지 않아 정확한 약동학적 파라미터의 산출에 제한이 있으나, 현 용법 유지 시 Therapeutic level에 이를 것으로 예상됩니다. 현 용법 유지 가능합니다.\n\n',
                                                       'rec2_공통': '2. 암모니아 농도를 포함한 간기능 및 임상상의 변화에 유의하시고, [f/u날짜(요일)] 투약 30분 전 채혈을 통한 TDM f/u하시어 용법의 적절성을 재확인하시기 바랍니다.',
                                                       },
                                       'Uppermargin': {
                                           'rec1_SS': '1. Upper margin of therapeutic level 입니다. 적극적인 증상 조절이 필요한 경우, 당분간 현 용법 유지 가능합니다.\n\n',
                                           'rec1_NSS': '1. 아직 항정상태에 도달하지 않아 정확한 약동학적 파라미터의 산출에 제한이 있으나, 현 용법 유지 시 Upper margin of therapeutic level에 이를 것으로 예상됩니다. 적극적인 증상 조절이 필요한 경우, 당분간 현 용법 유지 가능합니다.\n\n',
                                           'rec2_공통': '2. 독성증상 발현 예방을 위하여 용법 변경을 고려하시는 경우에는 다음 투약부터 [변경약물용법(용량,투약방식,투여간격)] 로 용법 변경 가능하며, 이 때 예상되는 항정상태에서의 농도는 아래와 같습니다.\n\n3. 암모니아 농도를 포함한 간기능 및 임상상의 변화에 유의하시고, [f/u날짜(요일)] 투약 30분 전 채혈을 통한 TDM f/u하시어 용법의 적절성을 재확인하시기 바랍니다.', },
                                       'Toxic': {'rec1_SS': '1. Toxic level입니다. ',
                                                 'rec1_NSS-nontoxic': '1. 아직 항정상태에 도달하지 않아 정확한 약동학적 파라미터의 산출에 제한이 있으나, 현 용법 유지 시 Toxic level에 도달할 것으로 예상됩니다. ',
                                                 'rec1_NSS-toxic': '1. 아직 항정상태에 도달하지 않아 정확한 약동학적 파라미터의 산출에 제한이 있으나, 현재 Toxic level입니다. ',
                                                 'rec2_용법변경': '용법 변경 권장합니다.\n\n2. 독성증상 발현 예방을 위하여 다음 투약부터 [변경약물용법(용량,투약방식,투여간격)] 로 용법 변경 가능하며, 이 때 예상되는 항정상태에서의 농도는 아래와 같습니다.\n\n3. 암모니아 농도를 포함한 간기능 및 임상상의 변화에 유의하시고, [f/u날짜(요일)] 투약 30분 전 채혈을 통한 TDM f/u하시어 용법의 적절성을 재확인하시기 바랍니다.',
                                                 'rec2_휴약후변경': '휴약 및 용법 변경 권장합니다.\n\n2. 독성 증상 발현 예방을 위하여 다음 예정된 투약 [기존예정된투약일(요일,시간)] 부터 휴약하시고 약물 농도가 충분히 감소할 것으로 예상되는 [변경용법적용날짜(요일,시간,예상농도)] 이후부터 [변경약물용법(용량,투약방식,투여간격)] 로 용법 변경 가능하며, 이 때 예상되는 항정상태에서의 농도는 아래와 같습니다.\n\n3. 암모니아 농도를 포함한 간기능 및 임상상의 변화에 유의하시고, [f/u날짜(요일)] 투약 30분 전 채혈을 통한 TDM f/u하시어 용법의 적절성을 재확인하시기 바랍니다.',
                                                 'rec2_휴약후f/u': '휴약 권장합니다.\n\n2. 암모니아 농도를 포함한 간기능 및 임상상의 변화에 유의하시고, 휴약을 유지하시다가, [f/u날짜(요일)] 정규채혈로 TDM 재의뢰하시어 재개 가능 여부를 재확인 하시기 바랍니다.',
                                                 },
                                       },
                               'GTM': {'Subtherapeutic': {'rec1_SS': '1. Subtherapeutic level입니다. 용법 변경 권장합니다.\n\n',
                                                          'rec1_NSS': '1. 아직 항정상태에 도달하지 않았으나 현 용법 유지시 Subtherapeutic level에 이를 것으로 예상됩니다. 용법 변경 권장합니다.\n\n',
                                                          'rec2_공통': '2. 적극적인 감염 조절을 위하여 다음 투약부터 [변경약물용법(용량,투약방식,투여간격)] 로 용법 변경 권장하며, 이 때 예상되는 항정상태에서의 농도는 아래와 같습니다.\n\n3. 신기능 및 임상상의 변화에 유의하시고, [f/u날짜(요일)] 투약 전 30분 및 투약 후 30분에 각각 1회씩 채혈을 통해 TDM f/u하시어 변경 용법의 적절성을 재확인하시기 바랍니다.',
                                                          },
                                       'Lowermargin': {
                                           'rec1_SS': '1. Lower margin of therapeutic Level 입니다. 감염 조절이 원활하다면 당분간 현 용법 유지 가능합니다.\n\n',
                                           'rec1_NSS': '1. 아직 항정상태에 도달하지 않았으나 현 용법 유지시 Lower margin of therapeutic level에 이를 것으로 예상됩니다. 감염 조절이 원활하다면 당분간 현 용법 유지 가능합니다.\n\n',
                                           'rec2_공통': '2. 적극적인 감염 조절을 위하여 용법 변경을 고려하시는 경우에는 다음 투약부터 [변경약물용법(용량,투약방식,투여간격)] 로 용법 변경 가능하며, 이 때 예상되는 항정상태에서의 농도는 아래와 같습니다.\n\n3. 신기능 및 임상상의 변화에 유의하시고, [f/u날짜(요일)] 투약 전 30분 및 투약 후 30분에 각각 1회씩 채혈을 통해 TDM f/u하시어 용법의 적절성을 재확인하시기 바랍니다.',
                                           },
                                       'LMofPeak': {
                                           'rec1_SS': '1. Lower margin of therapeutic peak level 입니다. 감염 조절이 원활하다면 당분간 현 용법 유지 가능합니다.\n\n',
                                           'rec1_NSS': '1. 아직 항정상태에 도달하지 않았으나 현 용법 유지시 Lower margin of therapeutic peak level에 이를 것으로 예상됩니다. 감염 조절이 원활하다면 당분간 현 용법 유지 가능합니다.\n\n',
                                           'rec2_공통': '2. 적극적인 감염 조절을 위하여 용법 변경을 고려하시는 경우에는 다음 투약부터 [변경약물용법(용량,투약방식,투여간격)] 로 용법 변경 가능하며, 이 때 예상되는 항정상태에서의 농도는 아래와 같습니다.\n\n3. 신기능 및 임상상의 변화에 유의하시고, [f/u날짜(요일)] 투약 전 30분 및 투약 후 30분에 각각 1회씩 채혈을 통해 TDM f/u하시어 용법의 적절성을 재확인하시기 바랍니다.',
                                           },
                                       'Therapeutic': {'rec1_SS': '1. Therapeutic level 입니다. 현 용법 유지 가능합니다.\n\n',
                                                       'rec1_NSS': '1. 아직 항정상태에 도달하지 않았으나 현 용법 유지시 Therapeutic level에 이를 것으로 예상됩니다. 현 용법 유지 가능합니다.\n\n',
                                                       'rec2_공통': '2. 신기능 및 임상상의 변화에 유의하시고, [f/u날짜(요일)] 투약 전 30분 및 투약 후 30분에 각각 1회씩 채혈을 통해 TDM f/u하시어 유지 용법의 적절성을 재확인하시기 바랍니다.',
                                                       },
                                       'Uppermargin': {
                                           'rec1_SS': '1. Upper margin of therapeutic level 입니다. 적극적인 감염 조절이 필요한 경우, 당분간 현 용법 유지 가능합니다.\n\n',
                                           'rec1_NSS': '1. 아직 항정상태에 도달하지 않았으나 현 용법 유지시 Upper margin of therapeutic level에 이를 것으로 예상되나 적극적인 증상 조절이 필요한 경우, 당분간 현 용법 유지 가능합니다.\n\n',
                                           'rec2_공통': '2. 독성증상 발현 예방을 위하여 용법 변경을 고려하시는 경우에는 다음 투약부터 [변경약물용법(용량,투약방식,투여간격)] 로 용법 변경 가능하며, 이 때 예상되는 항정상태에서의 농도는 아래와 같습니다.\n\n3. 신기능 및 임상상의 변화에 유의하시고, [f/u날짜(요일)] 투약 전 30분 및 투약 후 30분에 각각 1회씩 채혈을 통해 TDM f/u하시어 용법의 적절성을 재확인하시기 바랍니다.',
                                           },
                                       'UMofPeak': {
                                           'rec1_SS': '1. Upper margin of therapeutic peak level 입니다. 적극적인 감염 조절이 필요한 경우, 당분간 현 용법 유지 가능합니다.\n\n',
                                           'rec1_NSS': '1. 아직 항정상태에 도달하지 않았으나 현 용법 유지시 Upper margin of therapeutic peak level에 이를 것으로 예상되나 적극적인 증상 조절이 필요한 경우, 당분간 현 용법 유지 가능합니다.\n\n',
                                           'rec2_공통': '2. 독성증상 발현 예방을 위하여 용법 변경을 고려하시는 경우에는 다음 투약부터 [변경약물용법(용량,투약방식,투여간격)] 로 용법 변경 가능하며, 이 때 예상되는 항정상태에서의 농도는 아래와 같습니다.\n\n3. 신기능 및 임상상의 변화에 유의하시고, [f/u날짜(요일)] 투약 전 30분 및 투약 후 30분에 각각 1회씩 채혈을 통해 TDM f/u하시어 용법의 적절성을 재확인하시기 바랍니다.',
                                           },
                                       'Toxic': {'rec1_SS': '1. Toxic level입니다. ',
                                                 'rec1_NSS-nontoxic': '1. 아직 항정상태에 도달하지 않았으나 현 용법 유지시 Toxic level에 이를 것으로 예상됩니다. ',
                                                 'rec1_NSS-toxic': '1. 아직 항정상태에 도달하지 않았으나 현재 Toxic level입니다. ',
                                                 'rec2_용법변경': '용법 변경 권장합니다.\n\n2. 독성증상 발현 예방을 위하여 다음 투약부터 [변경약물용법(용량,투약방식,투여간격)] 로 용법 변경 가능하며, 이 때 예상되는 항정상태에서의 농도는 아래와 같습니다.\n\n3. 신기능 및 임상상의 변화에 유의하시고, [f/u날짜(요일)] 투약 전 30분 및 투약 후 30분에 각각 1회씩 채혈을 통해 TDM f/u하시어 용법 변경의 적절성을 재확인하시기 바랍니다.',
                                                 'rec2_휴약후변경': '휴약 및 용법 변경 권장합니다.\n\n2. 독성증상 발현 예방을 위하여 예정되어있는 다음 투약 [기존예정된투약일(요일,시간)] 부터 휴약하시고 약물 농도가 충분히 감소할 것으로 예상되는 [변경용법적용날짜(요일,시간,예상농도)] 이후부터 [변경약물용법(용량,투약방식,투여간격)] 로 용법 변경 가능하며, 이 때 예상되는 항정상태에서의 농도는 아래와 같습니다.\n\n3. 신기능 및 임상상의 변화에 유의하시고, [f/u날짜(요일)] 투약 전 30분 및 투약 후 30분에 각각 1회씩 채혈을 통해 TDM f/u하시어 용법 변경의 적절성을 재확인하시기 바랍니다.',
                                                 'rec2_휴약후f/u': '휴약 권장합니다.\n\n2. 신기능 및 임상상의 변화에 유의하시고, 휴약을 유지하시다가, [f/u날짜(요일)] 정규 채혈을 통한 TDM f/u하시어 투약 재개 시점 및 재개 용법을 재확인하시기 바랍니다.',
                                                 },
                                       },
                               }

        self.threshold_dict = {"AMK": {'toxic': 25.1, 'upper_margin': 24.9, 'lower_margin': 5.1, 'subthera': 4.9},
                               "VCM": {'toxic': 610, 'upper_margin': 590, 'lower_margin': 410, 'subthera': 395},
                               "DGX": {'toxic': 1.51, 'upper_margin': 1.49, 'lower_margin': 0.51, 'subthera': 0.49},
                               "GTM": {'toxic': 610, 'upper_margin': 590, 'lower_margin': 410, 'subthera': 395},
                               }


    def get_parameter_input(self):

        # fu_dt = (datetime.strptime(self.tdm_date, '%Y-%m-%d') + timedelta(days=2))

        weight = st.session_state['weight']
        drug = self.short_drugname_dict[st.session_state['drug']]
        age = st.session_state['age']
        round_num = 1 if drug != 'DGX' else 2

        self.ir_dict = dict()
        for k, v in self.ir_term_dict[drug].items():
            self.ir_dict[k] = st.session_state[k]
            if k in ('total_vd', 'vd_ss'):
                self.ir_dict['vd'] = float(round(st.session_state[k] / weight, 1))
            if (k == 'total_cl') and (drug == 'VCM'):
                self.ir_dict['cl'] = float(round(st.session_state[k] * 1000 / 60 / weight, 1))

        calc_text = ''
        drug_conc_text = ''

        if (drug == 'VCM'):
            if (age > 18):
                calc_text = f"Vd(L/kg) {self.ir_dict['vd']}\nVc {round(self.ir_dict['vc'],1)}\nCL(ml/min/kg) {self.ir_dict['cl']}\nCL(L/hr) {round(self.ir_dict['total_cl'], round_num)}\nt1/2(hr) {float(round(self.ir_dict['half_life'], 1))}\nVd ss {self.ir_dict['vd_ss']}"
            else:
                calc_text = f"Vd(L/kg) {self.ir_dict['vd']}\nCL(ml/min/kg) {self.ir_dict['cl']}\nCL(L/hr) {round(self.ir_dict['total_cl'], round_num)}\nt1/2(hr) {float(round(self.ir_dict['half_life'], 1))}\nVd ss {self.ir_dict['vd_ss']}"
            auc_val = round((self.ir_dict['adm_amount'] * (24 / self.ir_dict['adm_interval'])) / round(self.ir_dict['total_cl'], round_num), round_num)
            drug_conc_text = f"\n==========================================================================\n= Drug concentration ( Target : {self.tdm_target_txt_dict[drug]})\n1) 추정 Peak: {float(round(self.ir_dict['est_peak'], round_num))} ㎍/mL\n2) 추정 Trough: {float(round(self.ir_dict['est_trough'], round_num))} ㎍/mL\n3) 추정 AUC: {auc_val} mg*h/L\n\n"

            if (age > 18):
                pass
            else:
                pedi_type = ''
                if (age <= 6):
                    pedi_type = '영유아'
                elif (age > 6) and (age <= 12):
                    pedi_type = '소아'
                elif (age > 13):
                    pedi_type = '청소년'

                # print(
                #     f"\n\n### \n* 추가 유의사항 멘트 : {pedi_type}의 경우 성장에 따른 약동학 파라미터 변화의 폭이 크므로 신기능 및 임상상의 변화에 특히 유의하시고,\n\n")

        elif drug == 'DGX':

            calc_text = f"Vd(L/kg) {self.ir_dict['vd']}\nCL(L/hr) {round(self.ir_dict['total_cl'], 1)}\nt1/2(hr) {round(self.ir_dict['half_life'], 1)}"
            drug_conc_text = f"\n==========================================================================\n= Drug concentration ( Target : {self.tdm_target_txt_dict[drug]})\n1) 추정 Peak: {float(round(self.ir_dict['est_peak'], round_num))} ng/mL\n2) 추정 Trough: {float(round(self.ir_dict['est_trough'], round_num))} ng/mL\n\n"


        elif drug == 'AMK':
            calc_text = f"Vd(L/kg) {self.ir_dict['vd']}\nCL(L/hr) {round(self.ir_dict['total_cl'], round_num)}\nt1/2(hr) {round(self.ir_dict['half_life'], round_num)}"
            drug_conc_text = f"\n==========================================================================\n= Drug concentration ( Target : {self.tdm_target_txt_dict[drug]})\n1) 추정 Peak: {float(round(self.ir_dict['est_peak'], round_num))} ㎍/mL\n2) 추정 Trough: {float(round(self.ir_dict['est_trough'], round_num)) if self.ir_dict['est_trough'] >= 0.3 else '<0.3'} ㎍/mL\n\n"


        elif drug == 'GTM':
            calc_text = f"Vd(L/kg) {self.ir_dict['vd']}\nCL(L/hr) {round(self.ir_dict['total_cl'], round_num)}\nt1/2(hr) {round(self.ir_dict['half_life'], round_num)}"
            drug_conc_text = f"\n==========================================================================\n= Drug concentration ( Target : {self.tdm_target_txt_dict[drug]})\n1) 추정 Peak: {float(round(self.ir_dict['est_peak'], round_num))} ㎍/mL\n2) 추정 Trough: {float(round(self.ir_dict['est_trough'], round_num)) if self.ir_dict['est_trough'] >= 0.3 else '<0.3'} ㎍/mL\n\n"


        elif drug == 'VPA':
            calc_text = f"Vd(L/kg) {self.ir_dict['vd']}\nCL(L/hr) {round(self.ir_dict['total_cl'], round_num)}\nt1/2(hr) {round(self.ir_dict['half_life'], round_num)}"
            drug_conc_text = f"\n==========================================================================\n= Drug concentration ( Target : {self.tdm_target_txt_dict[drug]})\n1) 추정 Peak: {float(round(self.ir_dict['est_peak'], round_num))} ㎍/mL\n2) 추정 Trough: {float(round(self.ir_dict['est_trough'], round_num)) if self.ir_dict['est_trough'] >= 0.3 else '<0.3'} ㎍/mL\n\n"

        else: pass


        return calc_text + drug_conc_text

    def reflecting_parameters(self):

        parameter_input_text = self.get_parameter_input()

        prior_text = "Vd(L/kg)".join(st.session_state['first_draft'].split("Vd(L/kg)")[:-1])
        later_text = "= Interpretation : " + st.session_state['first_draft'].split("= Interpretation : ")[-1]
        st.session_state['first_draft'] = prior_text + parameter_input_text + later_text

    def offline_get_interpretation_and_recommendation_text(self, drug):

        self.define_ir_info()
        self.ir_dict = dict()

        self.wkday_dict = {0: '월', 1: '화', 2: '수', 3: '목', 4: '금', 5: '토', 6: '일'}
        fu_dt = (datetime.strptime(self.tdm_date, '%Y-%m-%d') + timedelta(days=3))

        print('PKS결과값을 입력합니다.\n')

        weight = self.pt_dict['weight']
        age = self.pt_dict['age']
        round_num = 1 if drug != 'DGX' else 2

        for k, v in self.ir_term_dict[drug].items():
            val_pass = False
            val_input = ''
            add_inst = ''

            if (k == 'vc') and (drug == 'VCM') and (age <= 18):
                continue

            while val_pass == False:
                raw_input = input(f'{v}{add_inst} : ').strip()
                val_pass, val_input = self.input_validation(key=k, value=raw_input)
            self.ir_dict[k] = val_input
            if k in ('total_vd', 'vd_ss'):
                self.ir_dict['vd'] = float(round(val_input / weight, round_num))
            if (k == 'total_cl') and (drug == 'VCM'):
                self.ir_dict['cl'] = float(round(val_input * 1000 / 60 / weight, round_num))

        calc_text=''
        drug_conc_text=''
        if (drug == 'VCM'):
            if (age > 18): calc_text = f"[PK Parameters]\n\nVd(L/kg) {self.ir_dict['vd']}\nVc {self.ir_dict['vc']}\nCL(ml/min/kg) {self.ir_dict['cl']}\nCL(L/hr) {round(self.ir_dict['total_cl'], round_num)}\nt1/2(hr) {float(round(self.ir_dict['half_life'], 1))}\nVd ss {self.ir_dict['vd_ss']}"
            else: calc_text = f"[PK Parameters]\n\nVd(L/kg) {self.ir_dict['vd']}\nCL(ml/min/kg) {self.ir_dict['cl']}\nCL(L/hr) {round(self.ir_dict['total_cl'], round_num)}\nt1/2(hr) {float(round(self.ir_dict['half_life'], 1))}\nVd ss {self.ir_dict['vd_ss']}"
            auc_val = round((self.ir_dict['adm_amount'] * (24 / self.ir_dict['adm_interval'])) / round(self.ir_dict['total_cl'], round_num), round_num)
            drug_conc_text = f"\n==========================================================================\n= Drug concentration ( Target : {self.tdm_target_txt_dict[self.pt_dict['drug']]})\n1) 추정 Peak: {float(round(self.ir_dict['est_peak'], round_num))} ㎍/mL\n2) 추정 Trough: {float(round(self.ir_dict['est_trough'], round_num))} ㎍/mL\n3) 추정 AUC: {auc_val} mg*h/L\n\n"

            if (age > 18): pass
            else:
                pedi_type = ''
                if (age <= 6): pedi_type = '영유아'
                elif (age > 6) and (age <= 12): pedi_type = '소아'
                elif (age > 13): pedi_type = '청소년'

                print(f"\n\n### \n* 추가 유의사항 멘트 : {pedi_type}의 경우 성장에 따른 약동학 파라미터 변화의 폭이 크므로 신기능 및 임상상의 변화에 특히 유의하시고,\n\n")

        elif drug == 'DGX':

            calc_text = f"[PK Parameters]\n\nVd(L/kg) {self.ir_dict['vd']}\nCL(L/hr) {round(self.ir_dict['total_cl'], round_num)}\nt1/2(hr) {round(self.ir_dict['half_life'], round_num)}"
            drug_conc_text = f"\n==========================================================================\n= Drug concentration ( Target : {self.tdm_target_txt_dict[self.pt_dict['drug']]})\n1) 추정 Peak: {float(round(self.ir_dict['est_peak'], round_num))} ng/mL\n2) 추정 Trough: {float(round(self.ir_dict['est_trough'], round_num))} ng/mL\n\n"


        elif drug == 'AMK':
            calc_text=f"[PK Parameters]\n\nVd(L/kg) {self.ir_dict['vd']}\nCL(L/hr) {round(self.ir_dict['total_cl'], round_num)}\nt1/2(hr) {round(self.ir_dict['half_life'], round_num)}"
            drug_conc_text=f"\n==========================================================================\n= Drug concentration ( Target : {self.tdm_target_txt_dict[self.pt_dict['drug']]})\n1) 추정 Peak: {float(round(self.ir_dict['est_peak'], round_num))} ㎍/mL\n2) 추정 Trough: {float(round(self.ir_dict['est_trough'], round_num)) if self.ir_dict['est_trough'] >= 0.3 else '<0.3'} ㎍/mL\n\n"


        elif drug == 'GTM':
            calc_text=f"[PK Parameters]\n\nVd(L/kg) {self.ir_dict['vd']}\nCL(L/hr) {round(self.ir_dict['total_cl'], round_num)}\nt1/2(hr) {round(self.ir_dict['half_life'], round_num)}"
            drug_conc_text=f"\n==========================================================================\n= Drug concentration ( Target : {self.tdm_target_txt_dict[self.pt_dict['drug']]})\n1) 추정 Peak: {float(round(self.ir_dict['est_peak'], round_num))} ㎍/mL\n2) 추정 Trough: {float(round(self.ir_dict['est_trough'], round_num)) if self.ir_dict['est_trough'] >= 0.3 else '<0.3'} ㎍/mL\n\n"


        elif drug == 'VPA':
            calc_text=f"[PK Parameters]\n\nVd(L/kg) {self.ir_dict['vd']}\nCL(L/hr) {round(self.ir_dict['total_cl'], round_num)}\nt1/2(hr) {round(self.ir_dict['half_life'], round_num)}"
            drug_conc_text=f"\n==========================================================================\n= Drug concentration ( Target : {self.tdm_target_txt_dict[self.pt_dict['drug']]})\n1) 추정 Peak: {float(round(self.ir_dict['est_peak'], round_num))} ㎍/mL\n2) 추정 Trough: {float(round(self.ir_dict['est_trough'], round_num)) if self.ir_dict['est_trough'] >= 0.3 else '<0.3'} ㎍/mL\n\n"

        print(calc_text+'\n ')
        print(drug_conc_text)
        result_text = calc_text+drug_conc_text
        return result_text

    def get_ir_text(self):

        drug = self.short_drugname_dict[st.session_state['drug']]

        self.ir_step_tups = ('ir_conc', 'ir_state', 'ir_method')

        self.ir_mediating_dict = dict([(irs, "") for irs in self.ir_step_tups])
        self.ir_result_dict = dict([(irs, "") for irs in self.ir_step_tups])

        for k in self.ir_step_tups:

            self.ir_mediating_dict[k] = st.session_state[k]

        iconc_str = self.interpretation_dict[self.ir_mediating_dict['ir_state'].split('-')[0]][self.ir_mediating_dict['ir_conc']]
        if self.ir_mediating_dict['ir_state']=='NSS-nontoxic':
            iconc_str+=' expected'

        rec1_key = f"rec1_{self.ir_mediating_dict['ir_state']}"
        rec2_key = f"rec2_{self.ir_mediating_dict['ir_method']}"

        rec1_str = self.ir_recomm_dict[drug][self.ir_mediating_dict['ir_conc']][rec1_key]
        rec2_str = self.ir_recomm_dict[drug][self.ir_mediating_dict['ir_conc']][rec2_key]

        prefix_str = ''
        if drug == 'DGX':
            prefix_str = '* Digoxin의 혈중 약물농도만으로는 약효 및 독성 발현 산출에 한계가 있으므로, 임상증상을 뒷받침하는 참고자료로 활용하시기 바랍니다.\n\n'

        self.ir_text = f"= Interpretation : \n{iconc_str}\n\n= Recommendation : \n{prefix_str}{rec1_str}{rec2_str}"
        # st.session_state['memo'] = str('농도는 아래와 같습니다.' in self.ir_text)
        if '농도는 아래와 같습니다.' in self.ir_text:
            if (drug == 'AMK') or (drug == 'GTM'): self.ir_text += f"\n\n= 변경시 예상농도 ( Target: {self.tdm_target_txt_dict[drug]})\n1) 변경시 예상 Peak :  ㎍/mL\n2) 변경시 예상 Trough :  ㎍/mL"
            elif drug == 'VCM': self.ir_text += f"\n\n= 변경시 예상농도 ( Target: {self.tdm_target_txt_dict[drug]})\n1) 변경시 예상 Peak :  ㎍/mL\n2) 변경시 예상 Trough :  ㎍/mL\n3) 변경시 예상 AUC :  mg*h/L"
            elif (drug == 'DGX') or (drug == 'VPA'): self.ir_text += f"\n\n= 변경시 예상농도 ( Target: {self.tdm_target_txt_dict[drug]})\n1) 변경시 예상 Peak :  ng/mL\n2) 변경시 예상 Trough :  ng/mL"
            else: pass

        self.ir_text = self.ir_text.replace('\n\n', '\n \n')
        return self.ir_text

    def reflecting_ir_text(self):

        ir_text = self.get_ir_text()

        prior_text = "= Interpretation : ".join(st.session_state['first_draft'].split("= Interpretation : ")[:-1])
        later_text = "문의사항은 다음의 전화번호로." + st.session_state['first_draft'].split("문의사항은 다음의 전화번호로.")[-1]
        st.session_state['first_draft'] = prior_text + ir_text + "\n\n\n" + later_text


    def offline_ir_text_generator(self, mode='manual', drug='', co_med_list=[]):

        ## 특별히 기입한 병용약물 없다면 -> 오더기록에서 있는지 확인 후 구성
        if len(co_med_list)==0:
            try: co_med_list = self.pt_dict['concomitant_medi'].split(', ')
            except: co_med_list = list()
        if len(co_med_list)==0:
            co_med_str = ''
        else:
            co_med_str = ''

        self.ir_drug_dict = self.ir_recomm_dict[drug]

        self.ir_text = ''
        if mode=='manual':

            # self.interpretation_dict.keys()
            self.ir_step_tups = ('ir_conc', 'ir_state', 'ir_method')
            self.ir_inform_str = {'ir_conc':"Interpretation Conclusion을", 'ir_state':"Steady State 여부를", 'ir_method':"용법 사용 여부를"}
            self.ir_flow_dict = dict([(irs, dict()) for irs in self.ir_step_tups])
            self.ir_mediating_dict = dict([(irs,"") for irs in self.ir_step_tups])
            self.ir_result_dict = dict([(irs, "") for irs in self.ir_step_tups])

            # k = 'ir_conc'
            # k = 'ir_state'
            # k = 'ir_method'
            # self.ir_mediating_dict['ir_conc'] = 'Subtherapeutic'
            for k in self.ir_step_tups:

                if k=='ir_conc':
                    self.ir_flow_dict[k] = list(self.ir_drug_dict.keys())
                elif k=='ir_state':
                    self.ir_flow_dict[k] = [s.split('_')[-1] for s in list(self.ir_drug_dict[self.ir_mediating_dict['ir_conc']].keys()) if s.split('_')[0]=='rec1']
                elif k=='ir_method':
                    self.ir_flow_dict[k] = [s.split('_')[-1] for s in list(self.ir_drug_dict[self.ir_mediating_dict['ir_conc']].keys()) if s.split('_')[0]=='rec2']

                val_pass = False
                val_input = ''
                while val_pass == False:
                    if len(self.ir_flow_dict[k])==1:
                        val_pass, val_input = True, 0
                    else:
                        print(f'[{self.ir_inform_str[k]} 선택하세요.]')
                        for inx, ir_state in enumerate(self.ir_flow_dict[k]):
                            print(f"{inx}. {ir_state}")

                        raw_input = input().strip()
                        val_pass, val_input = self.input_validation(key=k, value=raw_input, etc_info={f'{k}_list':self.ir_flow_dict[k]})
                    # val_input=0

                self.ir_mediating_dict[k] = self.ir_flow_dict[k][val_input]

            iconc_str = self.interpretation_dict[self.ir_mediating_dict['ir_state']][self.ir_mediating_dict['ir_conc']]

            rec1_key = f"rec1_{self.ir_mediating_dict['ir_state']}"
            rec2_key = f"rec2_{self.ir_mediating_dict['ir_method']}"

            rec1_str = self.ir_recomm_dict[drug][self.ir_mediating_dict['ir_conc']][rec1_key]
            rec2_str = self.ir_recomm_dict[drug][self.ir_mediating_dict['ir_conc']][rec2_key]

            prefix_str = ''
            if drug=='DGX':
                prefix_str = '* Digoxin의 혈중 약물농도만으로는 약효 및 독성 발현 산출에 한계가 있으므로, 임상증상을 뒷받침하는 참고자료로 활용하시기 바랍니다.\n\n'

            self.ir_text = f"= Interpretation : \n{iconc_str}\n\n= Recommendation : \n{prefix_str}{rec1_str}{rec2_str}"
            if '농도는 아래와 같습니다.' in self.ir_text:
                if (drug in ('AMK', 'GTM', 'VPA')): self.ir_text += f"\n\n= 변경시 예상농도 ( Target: {self.tdm_target_txt_dict[self.short_drugname_dict[st.session_state['drug']]]})\n1) 변경시 예상 Peak :  ㎍/mL\n2) 변경시 예상 Trough :  ㎍/mL"
                elif drug=='VCM': self.ir_text += f"\n\n= 변경시 예상농도 ( Target: {self.tdm_target_txt_dict[self.short_drugname_dict[st.session_state['drug']]]})\n1) 변경시 예상 Peak :  ㎍/mL\n2) 변경시 예상 Trough :  ㎍/mL\n3) 변경시 예상 AUC :  mg*h/L"
                elif (drug=='DGX') : self.ir_text += f"\n\n= 변경시 예상농도 ( Target: {self.tdm_target_txt_dict[self.short_drugname_dict[st.session_state['drug']]]})\n1) 변경시 예상 Peak :  ng/mL\n2) 변경시 예상 Trough :  ng/mL"
                else: pass
            self.ir_text = self.ir_text.replace('\n\n','\n \n')
            # print(self.ir_text)


        elif mode=='auto':
            pass
        print(self.ir_text)
        return self.ir_text


    def get_drug_concentration_text(self, drug):
        # drug = self.pt_dict['drug']
        self.concentration_text = ""
        self.concentration_dict = {"VCM":"Vanco", "DGX":"Digoxin", "AMK":"Amikacin", "VPA":"Valproic", "GTM":"Gentamicin" }
        dc_cols = ['date',self.concentration_dict[drug]]
        self.concentration_df = pd.DataFrame(columns=dc_cols)
        try:
            self.concentration_df = self.ldf[dc_cols].dropna().sort_values(['date',self.concentration_dict[drug]],ascending=False)
        except:
            return ''
        for inx, row in self.concentration_df.iloc[:2].iterrows():
            dc_frag = f"{row['date']} 00:00 {row[self.concentration_dict[drug]]}\n"
            self.concentration_text+=dc_frag

        return self.concentration_text

    def open_result_txt(self):
        open_result_txt(drug=self.pt_dict['drug'], name=self.pt_dict['name'], id=self.pt_dict['id'], tdm_date=self.pt_dict['tdm_date'].replace('-', ''))

    def offline_execution_main(self):

        import os
        try: project_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
        except: project_dir = os.path.abspath(os.path.dirname(os.path.dirname("__file__")))
        self.result_dir = f"{project_dir}/result"
        self.inputrecord_dir = f"{project_dir}/input_records"
        self.reply_text_saving_dir = f"{self.result_dir}/reply_text"


        def filepath_for_offline_test():
            result_dir = f"{project_dir}/result"
            resource_dir = f"{project_dir}/resource"
            inputfiles_dir = f"{project_dir}/input_files"
            inputrecord_dir = f"{project_dir}/input_records"
            lab_inputfile_path = f"{inputfiles_dir}/lab_input.xlsx"
            reply_text_saving_dir = f"{result_dir}/reply_text"

            for cdir in result_dir, resource_dir, inputfiles_dir, inputrecord_dir: check_dir(cdir)
            for cdir in lab_inputfile_path, reply_text_saving_dir: check_dir(cdir)

        filepath_for_offline_test()

        print('환자TDM 기본정보를 입력합니다.\n')
        # k, v = 'drug', '약물'
        for k, v in self.basic_pt_term_dict.items():
            val_pass=False
            val_input = ''
            add_inst = ''
            if k=='tdm_date':
                self.pt_dict[k] = datetime.today().strftime('%Y-%m-%d')
                self.tdm_date = self.pt_dict[k]
                self.prev_date = (datetime.strptime(self.tdm_date,'%Y-%m-%d') - timedelta(days=self.win_period)).strftime('%Y-%m-%d')
                continue
            elif k=='pedi': continue
            elif k=='height': add_inst='(cm)'
            elif k=='weight': add_inst='(kg)'
            while val_pass==False:
                # if self.raw_lab_input=='Y': raw_input = ''
                # else: raw_input = input(f'{v}{add_inst} : ').strip()
                raw_input = input(f'{v}{add_inst} : ').strip()
                val_pass, val_input = self.input_validation(key=k, value=raw_input)
            self.pt_dict[k] = val_input



        for k, v in self.additional_pt_term_dict[self.pt_dict['drug']].items():

            val_pass=False
            val_input = ''
            add_inst = ''
            add_msg = ''
            if k == 'hemodialysis':
                add_msg = f'\n<참고> 환자등록번호 : {self.pt_dict["id"]} / 이름 : {self.pt_dict["name"]}'
            elif k == 'consult':
                self.pt_dict[k] = self.parse_patient_history(hx_df=self.pt_hx_df, cont_type=k)
                continue

            elif k == 'order': add_msg = f'\n<참고> 환자등록번호 : {self.pt_dict["id"]} / 이름 : {self.pt_dict["name"]}'
            elif k == 'lab': add_msg = f'\n<참고> 환자등록번호 : {self.pt_dict["id"]} / 이름 : {self.pt_dict["name"]}'
            elif k=='echocardiography': add_msg = f'\n<참고> 환자등록번호 : {self.pt_dict["id"]} / 이름 : {self.pt_dict["name"]}'
            elif k == 'electroencephalography': add_msg = f'\n<참고> 환자등록번호 : {self.pt_dict["id"]} / 이름 : {self.pt_dict["name"]}'

            while val_pass==False:
                raw_input = input(f'{v}{add_inst} : {add_msg}').strip()
                val_pass, val_input = self.input_validation(key=k, value=raw_input)
                add_msg=''
            self.pt_dict[k] = val_input

    def input_validation(self, key, value, etc_info=dict()):
        if key == 'tdm_date':
            if type(value)==str: return True, value
            else: return False, value
        elif key=='id':
            if type(value)==str: return True, value.upper()
            else: return False, value
        elif key=='name':
            if type(value) == str: return True, value.upper()
            else: return False, value
        elif key=='sex':
            if value in ('M', 'F', 'm', 'f'):
                return True, value.upper()
            else:
                print(f'{self.basic_pt_term_dict[key]}을 (M, F) 중 하나로 입력해주세요.')
                return False, value
        elif key=='age':
            try:
                value = int(value)
                if value <= 18:
                    self.pt_dict['pedi'] = True
                    print('소아여부 : True')
                else: self.pt_dict['pedi'] = False
                return True, value
            except:
                print(f'{self.basic_pt_term_dict[key]}를 정수로 입력해주세요. / 현재입력값 : {value}')
                return False, value
        elif key in ('height', 'weight'):
            try:
                value = float(value)
                return True, value
            except:
                print(f'{self.basic_pt_term_dict[key]}를 실수로 입력해주세요. / 현재입력값 : {value}')
                return False, value
        elif key in ('half_life','vd_ss','total_vd','total_cl','vc','est_peak','est_trough', 'adm_amount', 'adm_interval'):
            try:
                value = float(value)
                return True, value
            except:
                print(f'{self.ir_term_dict[key]}를 실수로 입력해주세요. / 현재입력값 : {value}')
                return False, value
        elif key=='drug':
            drug_upc = value.upper()
            if drug_upc in self.drug_list:
                for k, v in self.basic_pt_term_dict.items():
                    self.pt_term_dict[k] = v
                for k, v in self.additional_pt_term_dict[drug_upc].items():
                    self.pt_term_dict[k] = v
                return True, drug_upc
            else:
                print(f'{self.basic_pt_term_dict[key]}을 {self.drug_list} 중 하나로 입력해주세요.')
                return False, value
        elif key=='history':
            if type(value) == str:
                ## raw data 저장
                self.input_record_dirname = f"{self.pt_dict['name']}({self.pt_dict['id']}){self.pt_dict['sex']}{self.pt_dict['age']}({self.pt_dict['tdm_date']})"
                check_dir_continuous(dir_list=[self.input_record_dirname], root_path=self.inputrecord_dir)
                input_record_filepath = f"{self.inputrecord_dir}/{self.input_record_dirname}/{key}_{self.pt_dict['name']}.txt"
                with open(input_record_filepath, "w", encoding="utf-8-sig") as f: f.write(value)

                self.pt_hx_raw = self.get_reduced_sentence(value)
                if self.pt_hx_raw=='': value=''
                else:
                    self.pt_hx_df = self.get_pt_hx_df(hx_str=self.pt_hx_raw)
                    value = self.parse_patient_history(hx_df=self.pt_hx_df, cont_type=key)
                    # value = self.parse_patient_history(hx_df=self.pt_hx_df, cont_type='consult')
                self.input_recording(key=key, value=value)
                return True, value
            else: return False, ''
        elif key=='hemodialysis':
            if type(value) == str:
                value = self.get_reduced_sentence(value)
                self.input_recording(key=key, value=value)
                return True, value
            else: return False, value
        elif key=='electroencephalography':
            if type(value) == str:
                ## raw data 저장
                input_record_filepath = f"{self.inputrecord_dir}/{self.input_record_dirname}/{key}_{self.pt_dict['name']}.txt"
                with open(input_record_filepath, "w", encoding="utf-8-sig") as f: f.write(value)

                value = self.get_parsed_eeg_result(eeg_result=value)
                self.input_recording(key=key, value=value)
                return True, value
            else: return False, value
        elif key=='echocardiography':
            if type(value) == str:
                ## raw data 저장
                input_record_filepath = f"{self.inputrecord_dir}/{self.input_record_dirname}/{key}_{self.pt_dict['name']}.txt"
                with open(input_record_filepath, "w", encoding="utf-8-sig") as f: f.write(value)

                value = self.get_parsed_echocardiography_result(echo_result=value)
                self.input_recording(key=key, value=value)
                return True, value
            else: return False, value
        elif key=='ecg':
            if type(value) == str:
                ## raw data 저장
                input_record_filepath = f"{self.inputrecord_dir}/{self.input_record_dirname}/{key}_{self.pt_dict['name']}.txt"
                with open(input_record_filepath, "w", encoding="utf-8-sig") as f: f.write(value)

                value = self.get_parsed_ecg_result(ecg_result=value)
                self.input_recording(key=key, value=value)
                return True, value
            else: return False, value
        elif key=='consult':
            print('참고(VCM, AMK, GTM : 감염내과_IMI / DGX : 순환기내과_IM / VPA : 신경과_NR')
            if type(value) == str:
                value = self.get_reduced_sentence(value)
                return True, value
            else: return False, value
        elif key == 'vs':
            # value = ''
            if type(value)==str:
                ## raw data 저장
                input_record_filepath = f"{self.inputrecord_dir}/{self.input_record_dirname}/{key}_{self.pt_dict['name']}.txt"
                with open(input_record_filepath, "w", encoding="utf-8-sig") as f: f.write(value)

                value = self.parse_vs_record(raw_vs=value)
                self.input_recording(key=key, value=value)
                return True, value
            else: return False, value
        elif key == 'lab':
            if type(value)==str:
                ## raw data 저장
                input_record_filepath = f"{self.inputrecord_dir}/{self.input_record_dirname}/{key}_{self.pt_dict['name']}.txt"
                with open(input_record_filepath, "w", encoding="utf-8-sig") as f: f.write(value)

                self.get_parsed_lab_df(value)

                return True, self.ldf

        elif key == 'order':
            ## raw data 저장
            input_record_filepath = f"{self.inputrecord_dir}/{self.input_record_dirname}/{key}_{self.pt_dict['name']}.txt"
            with open(input_record_filepath, "w", encoding="utf-8-sig") as f:
                f.write(value)

            # value = self.pt_dict['order']
            # value = ''

            value = self.parse_order_record(order_str=value)
            self.input_recording(key=key, value=value)
            return True, value

            # if len(value)==str: return True, value
            # else: return False, value
        elif key in ('ir_conc','ir_state', 'ir_method'):
            try:
                value = int(value)
                if value > len(etc_info[f'{key}_list']):
                    print(f"{len(etc_info[f'{key}_list'])} 보다 큰 정수는 선택지에 없습니다. / 현재입력값 : {value}")
                    return False, value
                else:
                    return True, value
            except:
                print(f'정수로 입력해주세요. / 현재입력값 : {value}')
                return False, value
        else:
            if type(value)==str:
                self.input_recording(key=key, value=value)
                return True, value.upper()
            else: return False, value

    def save_result_offline(self):
        file_name = f"{self.pt_dict['drug']}_{self.pt_dict['name']}_{self.pt_dict['id']}_{self.pt_dict['tdm_date'].replace('-', '')}.txt"
        file_path = f"{self.reply_text_saving_dir}/{file_name}"
        with open(file_path, "w", encoding="utf-8-sig") as f:
            f.write(self.file_content)