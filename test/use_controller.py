import os, sys

local_bakingsoda_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path = [local_bakingsoda_path] + sys.path
from utils.test_policy import load_policy_and_env, run_policy

env, get_action = load_policy_and_env('/home/tuyen/data/sac/sac_s0',deterministic=True)
run_policy(env, get_action,max_ep_len=1000,num_episodes=10,render=False)