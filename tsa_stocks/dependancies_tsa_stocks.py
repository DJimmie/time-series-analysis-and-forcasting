""" Dependancies supporting the tsa_stocks.py program"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels as sm 
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima_model import ARIMA
import statistics

import os
import sys
import logging

import json

import datetime as dt
from datetime import timedelta

sys.path.insert(0,"C:\\Users\\dowdj\\OneDrive\\Documents\\GitHub\\my-modules-and-libraries\\program_work_dir")  # Temporary. Used to help finish development of modules.
# import program_work_dir as pwd
from program_work_dir import *

sys.path.insert(0, "C:\\Users\\dowdj\\OneDrive\\Documents\\GitHub\\my-modules-and-libraries\\gui_maker")
from gui_build import *

sys.path.insert(0, "C:\\Users\\dowdj\\OneDrive\\Documents\\GitHub\\Stock-Data-Analysis\\sda_modules")
from source_stock_data import *
# import source_stock_data as ssd


def the_program_folder(x):
    config_parameters={'database server':{'sqlite_server':'N/A'}}

    client=ClientFolder(x,config_parameters)
    ini_file=f'c:/my_python_programs/{client}/{client}.ini'

    log_file=f'c:/my_python_programs/{client}/{client}_log.log'
    logging.basicConfig(filename=log_file, level=logging.INFO, filemode='w', format=' %(asctime)s -%(levelname)s - %(message)s')
    # logging.info('Start')

    return ini_file,log_file,client