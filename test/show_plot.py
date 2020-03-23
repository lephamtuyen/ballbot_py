import os, sys

local_bakingsoda_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path = [local_bakingsoda_path] + sys.path

from utils.plot import make_plots


make_plots(all_logdirs=["/home/tuyen/data/sac"], legend=None, xaxis='TotalEnvInteracts', values='EpLen', count=False,
               font_scale=1.5, smooth=1, select=None, exclude=None, estimator='mean')

