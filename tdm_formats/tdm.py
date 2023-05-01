import streamlit as st

class tdm():
    def __init__(self):
        self.common_vars()
        self.individual_vars()

    def common_vars(self):
        self.tdm_date = ''
        self.hospital = ''
        self.division = ''
        self.tdm_user = ''

    def individual_vars(self):
        pass

    def set_basic_info(self, tdm_date, hospital, division, tdm_user):
        self.tdm_date = tdm_date
        self.hospital = hospital
        self.division = division
        self.tdm_user = tdm_user

    def common_execution_flow(self):
        # st.title('TDM practice')
        # st.
        pass

    def individual_execution_flow(self):
        pass

    def exection_flow(self):
        self.common_execution_flow()
        self.individual_execution_flow()