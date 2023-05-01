import os, sys
try: project_dir = os.path.abspath(os.path.dirname(__file__))
except: project_dir = os.path.abspath(os.path.dirname("__file__"))
sys.path.append(project_dir)

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

from tdm_formats.tdm_snubh_cpt import *