import re
import pandas as pd
import numpy as np
from tdm_formats.tdm import *
from datetime import datetime, timedelta
import json

class snuh_cpt_tdm(tdm):
    def __init__(self):
        super().__init__()