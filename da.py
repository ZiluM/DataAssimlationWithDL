import sys,os
import yaml
import datetime as dt
import utils
import numpy as np
from dateutil.relativedelta import relativedelta
import torch


# print (sys.argv)
yml_path = sys.argv[1]




with open(yml_path, 'r') as stream:
    config = yaml.safe_load(stream)
sys.path.append(config['model_config'])
from myconfig1 import mypara
from Geoformer import Geoformer

logger = utils.get_logger()
config['logger'] = logger
config['obs_mean_length'] = int(config['obs_mean_length'])

logger.info("DA starts ======================")
job_path = config['save_path'] + config['job_name'] + '/'
if not os.path.exists(job_path) :
    os.mkdir(job_path)
logger.info("job path: " + job_path)
os.system(f'cp {yml_path} {job_path}')
logger.info(f"copy yml file: {yml_path} to job path: {job_path}")



mypara.output_length = config['obs_mean_length']

tmp = config['start_time']

config['my_para'] = mypara

config['start_time'] = dt.datetime(int(tmp[:4]),int(tmp[4:6]),int(tmp[6:8]),int(tmp[8:10]))
config['current_time'] = dt.datetime(int(tmp[:4]),int(tmp[4:6]),int(tmp[6:8]),int(tmp[8:10]))

config['Nino34'] = {'xa':[],'xp':[]}



all_obs_type = utils.get_obs_type2(config)

all_obs = utils.get_obs2(all_obs_type,config)
utils.save_obs(all_obs,config)

ens_xp = utils.init_ensemble(config,mypara) # Nens,12,lev,lat,lon


# print (ens_xp.shape)
Nobs = len(all_obs_type)
Nens = config['Nens']
# Nlev = mypara.lev_range[1] - mypara.lev_range[0]
Nlev = config['level_num']
Nlat = mypara.lat_range[1] - mypara.lat_range[0]
Nlon = mypara.lon_range[1] - mypara.lon_range[0]
# Model Init ======================
if config['forecast_model'] == 'dl':
    adr_model = config['model_path']
    mymodel = Geoformer(mypara).to(mypara.device)
    if torch.cuda.is_available():
        # mymodel = torch.nn.DataParallel(mymodel)
        mymodel.load_state_dict(torch.load(adr_model))
    else:
        mymodel.load_state_dict(torch.load(adr_model,map_location=torch.device('cpu')))
    mymodel.eval()
    function_prediction = utils.dpl_forcast
elif config['forecast_model'] == 'lim':
    mymodel = utils.LIM_Model(config)

    function_prediction = utils.lim_forecast
elif config['forecast_model'] == 'dl1':
    adr_model = config['model_path']
    mymodel = Geoformer(mypara).to(mypara.device)
    if torch.cuda.is_available():
        # mymodel = torch.nn.DataParallel(mymodel)
        mymodel.load_state_dict(torch.load(adr_model))
    else:
        mymodel.load_state_dict(torch.load(adr_model,map_location=torch.device('cpu')))
    mymodel.eval()
    offline_model = utils.Offline_Model(config)
    function_prediction = utils.dpl_forcast1
    mymodel = [mymodel,offline_model]
elif config['forecast_model'] == 'offline':
    # pass
    mymodel = utils.Offline_Model(config)
    # off
    function_prediction = utils.offline_forecast
elif config['forecast_model'] == 'offline1':
    # 
    mymodel = utils.Offline_Model(config)
    function_prediction = utils.offline_forecast1


elif config['forecast_model'] == 'lim1':
    mymodel = utils.LIM_Model1(config)
    function_prediction = utils.lim_forecast1

xa_ls = []

obs_idx = 0

# DA starts
for DA_t in range(0,config['DA_length'],config['obs_mean_length']):
    logger.info("DA Cycle: Current Time:" + str(config['current_time']))
    logger.info("Forecast Time:" + str(config['current_time'])) # current time after one predtion time 
    xp = function_prediction(ens_xp,mymodel,config,mypara) # Nens,
    obs_data = all_obs[obs_idx]
    obs_idx = obs_idx + 1
    logger.info("Assimilation Step:" + str(DA_t))
    logger.info("xp shape:" + str(xp.shape))
    utils.save_xp(xp,config)
    for i in range(Nobs):
        obs_type = all_obs_type[i]

        zp = utils.H(xp,obs_type,config) 

        xa = utils.enkf_update_array(xp.reshape(Nens,-1).T,obs_data[i],zp,obs_type['error_var'],config=config).T # Nens,-1
        xa = xa.reshape(Nens,config['obs_mean_length'],Nlev,Nlat,Nlon)

        xp = xa
    xp = utils.del_xp(config,ens_xp[:,-config['obs_mean_length']:],xa)
    logger.info("xp var:" + str(np.var(xp,axis=0).mean()))
    utils.save_xa(xa,config)
    
    ens_xp = np.concatenate([ens_xp[:,config['obs_mean_length']:],xp],axis=1)

    config['current_time'] = config['current_time'] + relativedelta(months=1) * config['obs_mean_length']

logger.info("DA ends ======================")


# if config['lite_post']:
    # pass
res = utils.lite_post(config)
utils.save_lite_post(config,res)
    







