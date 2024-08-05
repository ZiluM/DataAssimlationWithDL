import numpy as np
# import h5py
import os,sys
import xarray as xr
import logging
import pandas as pd
import datetime as dt
import pickle as pkl
from dateutil.relativedelta import relativedelta
import cftime
from torch.utils.data import IterableDataset
from torch.utils.data import Dataset
sys.path.append("/glade/work/zilumeng/3D_trans/Code")
from Geoformer import Geoformer
import torch
import h5py
sys.path.append("/glade/work/zilumeng/SSNLIM")
from slim import *
from EOF import EOF
# sys.path.append("/glade/work/zilumeng/3D_trans/Da/post")
# from putils import REAL_DATA,field_corr
# import putils.REAL_DATA as REAL_DATA

# from 

class REAL_DATA:
    def __init__(self,config):
        # obs_path = config['obs_path']
        mypara = config['my_para']
        needtauxy = mypara.needtauxy
        lon_range = mypara.lon_range
        lat_range = mypara.lat_range
        lev_range = mypara.lev_range
        address = config['true_path']
        data_in = xr.open_dataset(address)
        self.lev = data_in["lev"].values
        self.lat = data_in["lat"].values
        self.lon = data_in["lon"].values
        self.lev_range = lev_range
        self.lon_range = lon_range
        self.lat_range = lat_range

        temp = data_in["temperatureNor"][
            :,
            lev_range[0] : lev_range[1],
            lat_range[0] : lat_range[1],
            lon_range[0] : lon_range[1],
        ].values
        temp = np.nan_to_num(temp)
        temp[abs(temp) > 999] = 0
        if needtauxy:
            taux = data_in["tauxNor"][
                :,
                lat_range[0] : lat_range[1],
                lon_range[0] : lon_range[1],
            ].values
            taux = np.nan_to_num(taux)
            taux[abs(taux) > 999] = 0
            tauy = data_in["tauyNor"][
                :,
                lat_range[0] : lat_range[1],
                lon_range[0] : lon_range[1],
            ].values
            tauy = np.nan_to_num(tauy)
            tauy[abs(tauy) > 999] = 0
            # --------------
            self.dataX = np.concatenate(
                (taux[:, None], tauy[:, None], temp), axis=1
            )
            del temp, taux, tauy
        else:
            self.dataX = temp
        start_time = config['start_time']
        times = pd.date_range("1980-01-01", "2021-12-31", freq="MS")
        start_index = np.abs(times - start_time).argmin()
        self.dataX = self.dataX[start_index:config['DA_length']+start_index]
        # data = sd.load()
        data = data_in
        stdtemp = data["stdtemp"][mypara.lev_range[0] : mypara.lev_range[1]].values
        stdtemp = np.nanmean(stdtemp, axis=(1, 2))
        stdtaux = data["stdtaux"].values
        stdtaux = np.nanmean(stdtaux, axis=(0, 1))
        stdtauy = data["stdtauy"].values
        stdtauy = np.nanmean(stdtauy, axis=(0, 1))
        stds = np.concatenate((stdtaux[None], stdtauy[None], stdtemp), axis=0)
        config['obs_std'] = stds
        lon_nino_loc = mypara.lon_nino_relative
        lat_nino_loc = mypara.lat_nino_relative
        Nino34 = self.dataX[:,2,lat_nino_loc[0]:lat_nino_loc[1],lon_nino_loc[0]:lon_nino_loc[1]].mean(axis=(1,2))
        config['real_nino34'] = Nino34 * stds[2]

def field_corr(field1, field2):
    """
    field1: time, nspace
    field2: time, nspace
    """
    field1a = field1 - field1.mean(axis=0)
    field2a = field2 - field2.mean(axis=0)
    covar = np.einsum("ij...,ij...->j...", field1a, field2a) / (field1a.shape[0] - 1)  # covar:nspace
    corr = covar / np.std(field1a, axis=0) / np.std(field2a, axis=0)  # corr: nspace
    return corr.real

class Offline_Model():
    def __init__(self,config):
        self.config = config
        self.obs_mean_length = config['obs_mean_length']
        self.model_index = config['model_idx']
        self.bc_path = config['background_path']
        self.ens = config['Nens']
        self.config = config
    def prediction(self,xa,month,length,seed=1):
        config = self.config
        path = self.bc_path.format(self.model_index)
        bc = np.load(path)
        times = pd.date_range("1850-01-01", "2014-12-31", freq="MS")
        lon = config['lon']
        lat = config['lat']
        # lev = config['lev']
        lev = np.arange(config['level_num'])
        da = xr.DataArray(bc, coords=[times, lev, lat, lon], dims=["time", "lev", "lat", "lon"])
        months = []
        for i in range(self.obs_mean_length):
            if month + i > 12:
                # months.append(month + i - 12)
                month0 = month + i - 12
                year0 = 1
            else:
                # months.append(month + i)
                month0 = month + i
                year0 = 0
            da_sel = (da[da['time.month'] == month0])[year0:]
            da_sel = da_sel[:self.ens]
            months.append(da_sel.values)
        months = np.array(months) # obs_mean_length,ens,lev,lat,lon
        months = months.swapaxes(0,1) # ens,obs_mean_length,lev,lat,lon
        return months
        # month_da = month_da.values
        
        # self.path = config['offline_path']


def offline_forecast(xa,model,config,mypara=None):
    # pass
    # model = 
    return model.prediction(xa,config['current_time'].month,config['obs_mean_length'])

def offline_forecast1(xa,model,config,mypara=None):
    # pass
    # model = 
    return model.prediction(xa,1,config['obs_mean_length'])

class LIM_Model():
    """
    Linear inverse model for prediction
    """
    def __init__(self,config):
        with open(config['lim_path'],'rb') as f:
            self.model = pkl.load(f)
        with open(config['eof_path'],'rb') as f:
            self.eof = pkl.load(f)
        self.pc_num = config['pc_num']

    def prediction(self,xa,month,length,seed=1):
        """
        xa: Nens,1,lev,lat,lon
        month: current month (time after prediction)
        """
        month = month -1 if month > 1 else 12
        Nens = xa.shape[0]
        shape1 = xa.shape[1] # 1
        levs = xa.shape[2]
        lats = xa.shape[3]
        lons = xa.shape[4]
        xa1 = xa.reshape(Nens*shape1,levs,lats,lons)
        # to pcs
        psupc = self.eof.projection(xa1)[:self.pc_num,:].T # Nens*shape1,pcs
        if psupc.shape[0] != xa1.shape[0] or psupc.shape[1] != self.pc_num:
            print(psupc.shape)
            print(xa1.shape)
            raise ValueError("shape error")
        # to prediction
        preds_pc = self.model.noise_intergral(psupc,month0=month,length=length,seed= seed)[1:] # drop the first one (initial value), (length,nens,pc_num)
        preds_pc = np.swapaxes(preds_pc,0,1) # nens,length,pc_num
        # preds_pc = preds_pc.reshape(Nens,length,)
        if preds_pc.shape != (Nens,length,self.pc_num):
            print(preds_pc.shape)
            raise ValueError("preds_pc shape error")
        xp = self.eof.decoder1(preds_pc)
        if xp.shape != (Nens,length,levs,lats,lons):
            print(xp.shape)
            raise ValueError("xp shape error")
        return xp
        # if xp.shape[]
    
# def LIM_Model1
class LIM_Model1():
    """
    on CS Linear inverse model for prediction
    """
    def __init__(self,config):
        with open(config['lim_path'],'rb') as f:
            self.model = pkl.load(f)
        with open(config['eof_path'],'rb') as f:
            self.eof = pkl.load(f)
        self.pc_num = config['pc_num']

    def prediction(self,xa,month,length,seed=1):
        """
        xa: Nens,1,lev,lat,lon
        month: current month (time after prediction)
        """
        # month = month -1 if month > 1 else 12
        Nens = xa.shape[0]
        shape1 = xa.shape[1] # 1
        levs = xa.shape[2]
        lats = xa.shape[3]
        lons = xa.shape[4]
        xa1 = xa.reshape(Nens*shape1,levs,lats,lons)
        # to pcs
        psupc = self.eof.projection(xa1)[:self.pc_num,:].T # Nens*shape1,pcs
        pc_num = psupc.shape[-1]
        if psupc.shape[0] != xa1.shape[0] or psupc.shape[1] != self.pc_num:
            print(psupc.shape)
            print(xa1.shape)
            raise ValueError("shape error")
        # to prediction
        out_arr = np.zeros((length+1,Nens * shape1, self.pc_num))
        preds_pc = self.model.noise_integration(psupc,length=length+1,seed=seed,length_out_arr=out_arr)[1:] # drop the first one (initial value), (length,nens,pc_num)
        # print("preds_pc -1:" + str(preds_pc[-1]))
        preds_pc = np.swapaxes(preds_pc,0,1) # nens,length,pc_num

        # preds_pc = preds_pc.reshape(Nens,length,)
        if preds_pc.shape != (Nens,length,self.pc_num):
            print(preds_pc.shape)
            raise ValueError("preds_pc shape error")
        xp = self.eof.decoder1(preds_pc)
        if xp.shape != (Nens,length,levs,lats,lons):
            print(xp.shape)
            raise ValueError("xp shape error")
        return xp




class load_noise():
    """
    load noise for each month and each lead time
    """
    def __init__(self,config):
        self.config = config
        self.noise_path = config['noise_path']
        self.months = np.arange(1,13)
        name_ls = list(range(1000,44001,1000))
        name_ls.append(44804)
        self.name_ls = name_ls
        times = pd.date_range(start="1850-01",end="2014-12",freq="MS")[12:-20]
        months = times.month
        months = np.concatenate([months]*23,axis=0)
        self.months = months
    
    def load(self, month,max_lead,number):
        """
        month: current month
        load noise for each month and each lead time
        """
        # month = month + 1 if month < 12 else 1 # next month for noise save
        noise_ls = []
        name_ls = self.name_ls
        idxs = np.random.randint(0,len(name_ls)-1,size=2,)
        # print(idxs)
        idxs = [name_ls[idxs[i]] for i in range(2)]
        number1 = number // 2
        number2 = number - number1
        numbers = [number1,number2]
        noise_ls = []
        for idx,num in zip(idxs,numbers): 
            name = "noise_cmip6_{}.npy".format(idx)
            path = self.noise_path + name
            noise = np.load(path)
            # print(noise.shape)
            slice0 = slice(idx-1000,idx)
            months1 = self.months[slice0]
            label = (months1 == month)
            noise1 = noise[label]
            idx_noise = np.arange(0,noise1.shape[0])
            rd_idx_noise = np.random.choice(idx_noise,num,replace=False)
            noise1 = noise1[rd_idx_noise,:max_lead]
            noise_ls.append(noise1)
        noise_ls = np.concatenate(noise_ls,axis=0)
        return noise_ls
    
class load_noise2():
    """
    load noise for each lead time from EOF
    """
    def __init__(self,config):
        self.config = config
        self.noise_path = config['noise_path']
        self.eofs = []
        for i in range(0,12):
            with open(self.noise_path%i,'rb') as f:
                eof = pkl.load(f)
            self.eofs.append(eof)
    def load(self,month,max_lead,number,pc_num=100,correct=False,correct_amp=None):
        """
        month: current month
        number: number of noise
        """
        np.random.seed(self.config['seed'])
        patterns_ls = []
        # idx = lead_month - 1
        for idx in range(max_lead):
            eof = self.eofs[idx]
        # pcs = 
            pcs = np.random.randn(number,pc_num)
            patterns = eof.decoder1(pcs)
            if patterns.shape[0] != number:
                raise ValueError("patterns shape error, patterns shape[0]: %d"%patterns.shape[0])
            if correct:
                patterns = patterns * correct_amp
            patterns_ls.append(patterns)
        patterns_ls = np.array(patterns_ls) # lead_month,ens_number,lev,lat,lon
        patterns_ls = patterns_ls.swapaxes(0,1) # ens_number,lead_month,lev,lat,lon
        return patterns_ls




        
            




class make_dataset_ens():
    """
    online reading dataset
    """
    def __init__(self, mypara,ens_path,config):
        self.mypara = mypara
        self.config = config
        data_in = xr.open_dataset(ens_path)
        self.lev = data_in["lev"].values
        self.lat = data_in["lat"].values
        self.lon = data_in["lon"].values
        self.lev_range = mypara.lev_range
        self.lon_range = mypara.lon_range
        self.lat_range = mypara.lat_range
        self.input_length = mypara.input_length
        self.output_length = mypara.output_length
        self.all_group = mypara.all_group

        
        temp = data_in["temperatureNor"][
            :,
            :,
            mypara.lev_range[0] : mypara.lev_range[1],
            mypara.lat_range[0] : mypara.lat_range[1],
            mypara.lon_range[0] : mypara.lon_range[1],
        ].values
        config['lon'] = self.lon[self.lon_range[0] : self.lon_range[1]]
        config['lat'] = self.lat[self.lat_range[0] : self.lat_range[1]]
        config['lev'] = self.lev[self.lev_range[0] : self.lev_range[1]]
        temp = np.nan_to_num(temp)
        temp[abs(temp) > 999] = 0
        if mypara.needtauxy:
            config['logger'].info("loading tauxy...")
            taux = data_in["tauxNor"][
                :,
                :,
                mypara.lat_range[0] : mypara.lat_range[1],
                mypara.lon_range[0] : mypara.lon_range[1],
            ].values
            taux = np.nan_to_num(taux)
            taux[abs(taux) > 999] = 0
            tauy = data_in["tauyNor"][
                :,
                :,
                mypara.lat_range[0] : mypara.lat_range[1],
                mypara.lon_range[0] : mypara.lon_range[1],
            ].values
            tauy = np.nan_to_num(tauy)
            tauy[abs(tauy) > 999] = 0
            self.field_data = np.concatenate(
                (taux[:, :, None], tauy[:, :, None], temp), axis=2
            )
            del temp, taux, tauy
            config['level_num'] = 2 + self.lev_range[1] - self.lev_range[0]
        else:
            self.field_data = temp
            config['level_num'] = self.lev_range[1] - self.lev_range[0]
            del temp

    def load(self,month,ens_num,model_numbers = 20):
        data_time = pd.date_range("1850-01-01", "2014-12-31", freq="MS")
        input_length = self.input_length
        ens_num_current = 0
        ens = []
        ens_times = []
        # for i in range(self.field_data.shape[0]): # model numbers
        # i = 0
        for j in np.arange(model_numbers):
            model_data_indx = 0 
            while ens_num_current < ens_num:
                dataX = self.field_data[j, month + 12*(model_data_indx+1) - input_length : month + 12*(model_data_indx+1)]
                ens.append(dataX[np.newaxis])
                ens_times.append(data_time[month + 12*(model_data_indx+1) - input_length : month + 12*(model_data_indx+1)])
                ens_num_current += 1
                model_data_indx += 1
                if model_data_indx >= model_numbers:
                    continue
            # i += 1
        self.config['logger'].info("Ensemble Data Shape:" + str(np.concatenate(ens,axis=0).shape))
        self.config['logger'].info("Ensemble Data Time 1:" + str(np.concatenate(ens_times,axis=0)[:self.mypara.input_length]))
        return np.concatenate(ens,axis=0)






    # def __iter__(self):
    #     st_min = self.input_length - 1
    #     ed_max = self.field_data.shape[1] - self.output_length
    #     for i in range(self.all_group):
    #         rd_m = random.randint(0, self.field_data.shape[0] - 1)
    #         rd = random.randint(st_min, ed_max - 1)
    #         dataX = self.field_data[rd_m, rd - self.input_length + 1 : rd + 1]
    #         dataY = self.field_data[rd_m, rd + 1 : rd + self.output_length + 1]
    #         yield dataX, dataY


    def selectregion(self):
        return {
            "lon: {}E to {}E".format(
                self.lon[self.lon_range[0]],
                self.lon[self.lon_range[1] - 1],
            ),
            "lat: {}S to {}N".format(
                self.lat[self.lat_range[0]],
                self.lat[self.lat_range[1] - 1],
            ),
            "temp lev: {}m to {}m".format(
                self.lev[self.lev_range[0]], self.lev[self.lev_range[1] - 1]
            ),
        }


def get_logger():
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def get_obs_type(config):
    # get the observation
    # lats = []
    start_time = config['start_time']
    locs = config['obs_locs']
    Nobs = config['Nobs']
    all_obs = []
    for ob in range(Nobs):
        ob_dict = {}
        ob_dict['variable'] = 'tos'
        ob_dict['lat'] = float(locs[ob][1])
        ob_dict['lon'] = float(locs[ob][0])
        ob_dict['error_var'] = config['obs_noise']
        ob_dict['obs_mean_length'] = int(config['obs_mean_length'])
        all_obs.append(ob_dict)
    return all_obs


def get_obs_type2(config):
    with open(config['obs_locs'],'rb') as f:
        obs_data = pkl.load(f)
    start_time = config['start_time']
    Nobs = len(obs_data)
    config['Nobs'] = Nobs
    all_obs = []
    for ob in range(Nobs):
        ob_dict = {}
        ob_dict['variable'] = 'tos'
        ob_dict['lat'] = float(obs_data[ob]['lat'].values)
        ob_dict['lon'] = float(obs_data[ob]['lon'].values)
        ob_dict['error_var'] = config['obs_noise']
        ob_dict['obs_mean_length'] = int(config['obs_mean_length'])
        all_obs.append(ob_dict)
    return all_obs

def get_obs2(types,config):
    # get the observation
    obs_path = config['obs_path']
    data_times = pd.date_range("1980-01-01", "2021-12-31", freq="MS")
    current_time = config['start_time']
    with open(config['obs_locs'],'rb') as f:
        obs_data = pkl.load(f)
    config['logger'].info("OBS Data loaded Start:" + str(current_time))
    config['obs_std'] = [1,2,3] # taux, tauy, temp
    obs_ls = []
    for t in range(config['DA_length']):
        obs_current = []
        for idx,type in enumerate(types):
            lat = type['lat']
            lon = type['lon']
            obs = obs_data[idx].loc[current_time]
            obs_current.append(obs)
        obs_ls.append(obs_current)
        current_time += relativedelta(months=1)
    
    config['logger'].info("OBS Data loaded End:" + str(current_time))
    config['logger'].info("OBS before mean" + str(np.array(obs_ls).shape))

    res = np.array(obs_ls)
    mean_res = []
    for i in range(0,res.shape[0],config['obs_mean_length']):
        mean_res.append(res[i:i+config['obs_mean_length']].mean(axis=0) )
    mean_res = np.array(mean_res) # shape: obs time after mean_length, Nobs
    # add noise
    variance = mean_res.var(axis=0) # get variance for each obs
    R = variance * (config['obs_noise'] ** 2)
    noise = np.random.randn(mean_res.shape[0],mean_res.shape[1]) * np.sqrt(R) # noise for each obs
    for idx, type in enumerate(types):
        type['error_var'] = R[idx]
    if bool(config['add_noise']): # add noise to obs
        mean_res = mean_res + noise
    # mean_res = mean_res + noise
    config['logger'].info("noise shape:" + str(noise.shape))
    config['logger'].info("R:" + str(R))
    config['logger'].info("OBS Data Shape:" + str(mean_res.shape))
    return mean_res



def get_obs(types,config):
    # get the observation
    obs_path = config['obs_path']
    mypara = config['my_para']
    # data = sd.load()
    data = xr.open_dataset(obs_path)
    stdtemp = data["stdtemp"][mypara.lev_range[0] : mypara.lev_range[1]].values
    stdtemp = np.nanmean(stdtemp, axis=(1, 2))
    stdtaux = data["stdtaux"].values
    stdtaux = np.nanmean(stdtaux, axis=(0, 1))
    stdtauy = data["stdtauy"].values
    stdtauy = np.nanmean(stdtauy, axis=(0, 1))
    stds = np.concatenate((stdtaux[None], stdtauy[None], stdtemp), axis=0)
    config['obs_std'] = stds
    data_times = pd.date_range("1980-01-01", "2021-12-31", freq="MS")
    sst = data['temperatureNor'].loc[dict(lev=5)]
    sst = sst.rename({'n_mon': 'time'})
    sst['time'] = data_times
    current_time = config['start_time']
    config['logger'].info("OBS Data loaded Start:" + str(current_time))
    obs_ls = []

    for t in range(config['DA_length']):
        obs_current = []
        for type in types:
            lat = type['lat']
            lon = type['lon']
            obs = sst.sel(time=current_time, lat=lat, lon=lon, method='nearest')
            obs_current.append(obs.values)
        obs_ls.append(obs_current)
        current_time += relativedelta(months=1)
        # print(current_time)
    config['logger'].info("OBS Data loaded End:" + str(current_time))
    config['logger'].info("OBS before mean" + str(np.array(obs_ls).shape))
    # print(np.array(obs_ls).shape)
    res = np.array(obs_ls)
    mean_res = []
    for i in range(0,res.shape[0],config['obs_mean_length']):
        mean_res.append(res[i:i+config['obs_mean_length']].mean(axis=0) )
    mean_res = np.array(mean_res)
    config['logger'].info("OBS Data Shape:" + str(mean_res.shape))
    return mean_res


def init_ensemble(config,mypara):
    config['logger'].info("Ensemble Data loaded Start")
    ens_num = config['Nens'] 
    # slim_path = config['slim_path']
    ens_path = config['ens_path']
    # data = xr.open_dataset(ens_path)
    month = config['start_time'].month - 1
    # config['mypara'] = 
    # mypara = config['mypara']
    dataset = make_dataset_ens(mypara,ens_path,config)
    ens = dataset.load(month,ens_num)
    config['logger'].info("Ensemble Data loaded End")
    return ens


def dpl_forcast(xa,model,config,mypara):
    max_num = config['max_num']
    ens_num = xa.shape[0]
    model.eval()
    amp = float(config['noise_amp'])
    # if ens_num > max_num:
    loop_pred  = ens_num // max_num if ens_num % max_num == 0 else ens_num // max_num + 1
    mon = config['current_time'].month # current month after predtion time 
    # for lead in range(config['obs_mean_length']):
    #     # xa_current
    #     noise = np.load(config['noise_path'].format(lead,mon)) # mon: 
    # noise_dc = load_noise2(config)
    # noises = noise_dc.load(month=mon,max_lead=config['obs_mean_length'],number = ens_num) # Nens,obs_mean_length,lev,lat,lon
    noise_dc = load_noise(config)
    noises = noise_dc.load(mon,config['obs_mean_length'],ens_num)
    config['logger'].info("Noise Shape:" + str(noises.shape))
    for num in range(loop_pred): # loop for prediction because of the memory limit
        xa_current = xa[num*max_num:(num+1)*max_num]
        xa_current = torch.from_numpy(xa_current).float().to(mypara.device)
        out_var = model(
        xa_current,
        predictand=None,
        train=False,
        )
        out_var = out_var.cpu().detach().numpy()
        del xa_current
        if num == 0:
            xp = out_var
        else:
            xp = np.concatenate([xp,out_var],axis=0)
        del out_var
    if np.abs(config['cov_inf'] - 0 ) > 1e-6:
        xpm = np.mean(xp,axis=0) # mean on ens 
        xpa = xp - xpm # anomaly
        xpa = xpa * config['cov_inf'] # inflate
        xp = xpm + xpa # add mean
    xp = xp + noises * amp
    
    # config['logger'].info("Deep Learning Forecast Shape:" + str(xp.shape))
    # adr_model = "model/Geoformer_beforeTrans.pkl"
    # out_var = model(
    # torch.from_numpy(xa).float().to(mypara.device),
    # predictand=None,
    # train=False,
    # )
    return xp


def dpl_forcast1(xa,model,config,mypara):
    max_num = config['max_num']
    ens_num = xa.shape[0]
    model,offline_model = model
    # amp = float(config['noise_amp'])
    clim_alpha = config['clim_alpha']
    # if ens_num > max_num:
    loop_pred  = ens_num // max_num if ens_num % max_num == 0 else ens_num // max_num + 1
    mon = config['current_time'].month # current month after predtion time 
    # for lead in range(config['obs_mean_length']):
    #     # xa_current
    #     noise = np.load(config['noise_path'].format(lead,mon)) # mon: 
    # noise_dc = load_noise2(config)
    # noises = noise_dc.load(month=mon,max_lead=config['obs_mean_length'],number = ens_num) # Nens,obs_mean_length,lev,lat,lon
    noise_dc = load_noise(config)
    noises = noise_dc.load(mon,config['obs_mean_length'],ens_num)
    config['logger'].info("Noise Shape:" + str(noises.shape))
    for num in range(loop_pred): # loop for prediction because of the memory limit
        xa_current = xa[num*max_num:(num+1)*max_num]
        xa_current = torch.from_numpy(xa_current).float().to(mypara.device)
        out_var = model(
        xa_current,
        predictand=None,
        train=False,
        )
        out_var = out_var.cpu().detach().numpy()
        del xa_current
        if num == 0:
            xp = out_var
        else:
            xp = np.concatenate([xp,out_var],axis=0)
        del out_var
    if np.abs(config['cov_inf'] - 0 ) > 1e-6:
        xpm = np.mean(xp,axis=0) # mean on ens 
        xpa = xp - xpm # anomaly
        xpa = xpa * config['cov_inf'] # inflate
        xp = xpm + xpa # add mean
    else:
        pass
    clim = offline_model.prediction(xa,config['current_time'].month,config['obs_mean_length'])
    
    xp = clim_alpha * xp + (1-clim_alpha) * clim
    
    # config['logger'].info("Deep Learning Forecast Shape:" + str(xp.shape))
    # adr_model = "model/Geoformer_beforeTrans.pkl"
    # out_var = model(
    # torch.from_numpy(xa).float().to(mypara.device),
    # predictand=None,
    # train=False,
    # )
    return xp

# def 

def lim_forecast(xa,model: LIM_Model,config,mypara=None):
    xa = xa[:,[-1]] # select the last month for prediction
    month = config['current_time'].month
    length = config['obs_mean_length']
    xp = model.prediction(xa,month,length,seed=1)
    return xp

def lim_forecast1(xa,model: LIM_Model,config,mypara=None):
    xa = xa[:,[-1]] # select the last month for prediction
    month = config['current_time'].month
    length = config['obs_mean_length']
    xp = model.prediction(xa,month,length,seed=1)
    return xp

# def Offline_Model






def specific_loc_indx(lats,lons,lat,lon):
    """
    lats: (Ny,)
    lons: (Nx,)
    lat: float
    lon: float
    return: i: index of lon
            j: index of lat
    """
    lat_abs = np.abs(lats-lat)
    lon_abs = np.abs(lons-lon)
    return np.argmin(lon_abs) , np.argmin(lat_abs)

def H(ens_data, obs_type, config):
    """
    ens_data: (Nens, Nx...)
    obs_type : attributes of the observation
    config: config file
    """
    # sst_svd = 
    # print(sst_field.shape)
    lon = obs_type['lon']
    lat = obs_type['lat']
    # lons = np.arange(0,360,2)
    # lats = np.arange(-89,89+2,2)
    lons = config['lon']
    lats = config['lat']
    i,j = specific_loc_indx(lats,lons,lat,lon)
    # print(i,j)
    if config['my_para'].needtauxy:
        z_obs = ens_data[:,:,2,j,i].mean(axis=1)
    else:
        z_obs = ens_data[:,:,0,j,i].mean(axis=1) # 
    return z_obs


def enkf_update_array(Xb, obvalue, Ye, ob_err,config, loc=None, debug=False):
    """ Function to do the ensemble square-root filter (EnSRF) update

    (ref: Whitaker and Hamill, Mon. Wea. Rev., 2002)

    Originator:
        G. J. Hakim, with code borrowed from L. Madaus Dept. Atmos. Sciences, Univ. of Washington

    Revisions:
        1 September 2017: changed varye = np.var(Ye) to varye = np.var(Ye,ddof=1) for an unbiased calculation of the variance. (G. Hakim - U. Washington)

    Args:
        Xb: background ensemble estimates of state (Nx x Nens)
        obvalue: proxy value
        Ye: background ensemble estimate of the proxy (Nens x 1)
        ob_err: proxy error variance
        loc: localization vector (Nx x 1) [optional]
        inflate: scalar inflation factor [optional]
    """

    # Get ensemble size from passed array: Xb has dims [state vect.,ens. members]
    Nens = Xb.shape[1]

    # ensemble mean background and perturbations
    xbm = np.mean(Xb,axis=1)
    Xbp = np.subtract(Xb,xbm[:,None])  # "None" means replicate in this dimension

    # ensemble mean and variance of the background estimate of the proxy
    mye   = np.mean(Ye)
    varye = np.var(Ye,ddof=1)

    # lowercase ye has ensemble-mean removed
    ye = np.subtract(Ye, mye)

    # innovation
    try:
        innov = obvalue - mye
    except:
        print('innovation error. obvalue = ' + str(obvalue) + ' mye = ' + str(mye))
        print('returning Xb unchanged...')
        return Xb

    # innovation variance (denominator of serial Kalman gain)
    kdenom = (varye + ob_err)

    # numerator of serial Kalman gain (cov(x,Hx))
    kcov = np.dot(Xbp,np.transpose(ye)) / (Nens-1)

    # Option to inflate the covariances by a certain factor
    #if inflate is not None:
    #    kcov = inflate * kcov # This implementation is not correct. To be revised later.

    # Option to localize the gain
    if loc is not None:
        kcov = np.multiply(kcov,loc)

    # Kalman gain
    kmat = np.divide(kcov, kdenom)

    # update ensemble mean
    xam = xbm + np.multiply(kmat,innov)

    # update the ensemble members using the square-root approach
    beta = 1./(1. + np.sqrt(ob_err/(varye+ob_err)))
    kmat = np.multiply(beta,kmat)
    ye   = np.array(ye)[np.newaxis]
    kmat = np.array(kmat)[np.newaxis]
    Xap  = Xbp - np.dot(kmat.T, ye)

    # full state
    Xa = np.add(xam[:,None], Xap)

    # if masked array, making sure that fill_value = nan in the new array
    if np.ma.isMaskedArray(Xa): np.ma.set_fill_value(Xa, np.nan)

    # Return the full state
    return Xa


def rescale(xp,config):
    """
    xp: Nens,1,lev,lat,lon
    """
    xp = xp * config['obs_std'][None,None,:,None,None]
    return xp


def save_xa(xa,config):
    mean_time = config['obs_mean_length']

    path = config['save_path'] + config['job_name'] + '/'
    if not os.path.exists(path) :
        os.mkdir(path)
    file_name = path + 'xa_' + config['current_time'].strftime("%Y%m%d%H") + '.nc'
    file_name_mean = path + 'mxa_' + config['current_time'].strftime("%Y%m%d%H") + '.nc'
    # hf = h5py.File(file_name,'r+')
    da = xr.DataArray(xa.swapaxes(0,1),
                      coords={'time':[config['current_time'] + relativedelta(months=1) * i for i in range(0,mean_time)],
                              'ens':np.arange(config['Nens']),
                              'lev':np.arange(config['level_num']),
                              'lat':config['lat'],
                              'lon':config['lon']})
    dam = da.mean(dim='ens')
    Nino34 = da.loc[dict(lat=slice(-5,5),lon=slice(190,240),lev=2)].mean(dim=['lat','lon'])
    config['Nino34']['xa'].append(Nino34)
    ds = xr.Dataset({'xa':da})
    dsm = xr.Dataset({'xa':dam})
    ds.attrs['config'] = str(config)
    dsm.attrs['config'] = str(config)
    ds.to_netcdf(file_name)
    dsm.to_netcdf(file_name_mean)


    config['logger'].info("Saving Xa Data to:" + file_name)
    config['logger'].info("Saving Xa Mean Data to:" + file_name_mean)

def save_xp(xp,config):
    mean_time = config['obs_mean_length']
    path = config['save_path'] + config['job_name'] + '/'
    if not os.path.exists(path) :
        os.mkdir(path)
    file_name = path + 'xp_' + config['current_time'].strftime("%Y%m%d%H") + '.nc'
    file_name_mean = path + 'mxp_' + config['current_time'].strftime("%Y%m%d%H") + '.nc'
    # hf = h5py.File(file_name,'r+')
    da = xr.DataArray(xp.swapaxes(0,1),
                      coords={'time':[config['current_time'] + relativedelta(months=1) * i for i in range(0,mean_time)],
                              'ens':np.arange(config['Nens']),
                            #   'mean_length':np.arange(config['obs_mean_length']),
                              'lev':np.arange(config['level_num']),
                              'lat':config['lat'],
                              'lon':config['lon']})
    dam = da.mean(dim='ens')
    ds = xr.Dataset({'xp':da})
    dsm = xr.Dataset({'xp':dam})

    Nino34 = da.loc[dict(lat=slice(-5,5),lon=slice(190,240),lev=2)].mean(dim=['lat','lon'])
    config['Nino34']['xp'].append(Nino34)


    ds.attrs['config'] = str(config)
    dsm.attrs['config'] = str(config)

    ds.to_netcdf(file_name)
    dsm.to_netcdf(file_name_mean)

    config['logger'].info(config['current_time'].strftime("%Y%m%d%H") + "Saving Xp Data to:" + file_name)
    config['logger'].info(config['current_time'].strftime("%Y%m%d%H") + "Saving Xp Mean Data to:" + file_name_mean)


def save_obs(obs,config):
    path = config['save_path'] + config['job_name'] + '/'
    if not os.path.exists(path) :
        os.mkdir(path)
    file_name = path + 'obs' +  '.npy'
    # hf = h5py.File(file_name,'r+')
    np.save(file_name,np.array(obs))
    config['logger'].info("Saving Obs Data to:" + file_name)

def del_xp(config,xp,xa):
    """
    xp: prior before assimilation
    xa: posterior
    config: config file
    return: new xp for the next cycle
    """
    alpha = config['Ralpha']
    xpm = np.mean(xp,axis=0)
    xpa = xp - xpm
    xam = np.mean(xa,axis=0)
    xaa = xa - xam
    # xpa = 
    new_xp = xam + alpha * xpa + (1-alpha) * xaa
    return new_xp


def ensemble_calib_ratio(xp,real):
    ave_squ_err = (xp.mean(axis=1) - real) ** 2
    # sigma = 
    sigma = np.var(xp,axis=1)
    ratio = ave_squ_err / sigma
    mean_ratio = np.mean(ratio,axis=0)
    return mean_ratio


def lite_post(config):
    """
    lite post process
    """
    # get the observation
    obs_path = config['obs_path']
    mypara = config['my_para']
    path = config['save_path'] + config['job_name'] + '/'
    # real_data = xr.
    # data = xr.open_dataset(obs_path)
    # stdtemp = data["stdtemp"][mypara.lev_range[0] : mypara.lev_range[1]].values
    # stdtemp = np.nanmean(stdtemp, axis=(1, 2))
    # stdtaux = data["stdtaux"].values
    # stdtaux = np.nanmean(stdtaux, axis=(0, 1))
    # stdtauy = data["stdtauy"].values
    # stdtauy = np.nanmean(stdtauy, axis=(0, 1))
    # stds = np.concatenate((stdtaux[None], stdtauy[None], stdtemp), axis=0)
    # config['obs_std'] = stds
    data_times = pd.date_range("1980-01-01", "2021-12-31", freq="MS")
    config['end_time'] = config['start_time'] + relativedelta(months=config['DA_length'])
    # time_slice = slice(config['start_time'],config['end_time'])
    # sst = data['temperatureNor'].loc[dict(lev=5)][:,mypara.lev_range[0] : mypara.lev_range[1]]
    # sst = sst.rename({'n_mon': 'time'})
    # sst['time'] = data_times
    # sst = sst.loc[time_slice]
    # /glade/work/zilumeng/3D_trans/Da/res01/dl_01_covinf1.2/mxa_1980100100.nc
    real_data = REAL_DATA(config).dataX * config['obs_std'][:,None,None]
    Nino34_real = config['real_nino34'] 
    # sst = sst
    # Nino34_real = sst.loc[dict(lat=slice(-5,5),lon=slice(190,240))].mean(dim=['lat','lon'])  # real Nino34
    xa_Nino34 = xr.concat(config['Nino34']['xa'],dim='time') * config['obs_std'][2]
    xp_Nino34 = xr.concat(config['Nino34']['xp'],dim='time') * config['obs_std'][2]
    xa_Nino34_ds = xr.Dataset({'xa_Nino34':xa_Nino34})
    xp_Nino34_ds = xr.Dataset({'xp_Nino34':xp_Nino34})
    xa_Nino34_ds.to_netcdf(path + 'xa_Nino34.nc')
    xp_Nino34_ds.to_netcdf(path + 'xp_Nino34.nc')
    stds = config['obs_std']
    xam_Nino34 = xa_Nino34.mean(dim='ens')
    xpm_Nino34 = xp_Nino34.mean(dim='ens')
    ecr_p = ensemble_calib_ratio(xp_Nino34,Nino34_real)
    ecr_a = ensemble_calib_ratio(xa_Nino34,Nino34_real)
    xa_Nino34_corr = np.corrcoef(Nino34_real,xam_Nino34)[0,1]
    xa_Nino34_rmse = np.sqrt(np.mean((Nino34_real - xam_Nino34)**2))
    xp_Nino34_corr = np.corrcoef(Nino34_real,xpm_Nino34)[0,1]
    xp_Nino34_rmse = np.sqrt(np.mean((Nino34_real - xpm_Nino34)**2))
    # ==============================
    # open the mean file
    xa_mean = (xr.open_mfdataset(path + 'mxa_*' + '.nc')['xa'].chunk({"time":-1}) * stds[:,None,None] ).to_numpy()
    xa_sst = xa_mean[:,2]
    sst_real = real_data[:,2]
    xa_sst_mean = np.nanmean(xa_sst,axis=(1,2))
    sst_real_mean = np.nanmean(sst_real,axis=(1,2))
    sst_mean_corr = np.corrcoef(sst_real_mean,xa_sst_mean)[0,1]
    sst_mean_rmse = np.sqrt(np.mean((sst_real_mean - xa_sst_mean)**2))
    sst_sum_rmse = np.sqrt(np.nanmean((sst_real - xa_sst)**2))
    # sst corr mean
    corr_sst = field_corr(xa_sst,sst_real)
    sst_corr_mean = np.nanmean(corr_sst)
    # ==============================
    wind_stress = xa_mean[:,:2]
    ws_real = real_data[:,:2]
    wind_stress_mean = np.nanmean(wind_stress,axis=(1,2,3))
    wind_stress_real_mean = np.nanmean(ws_real,axis=(1,2,3))
    wind_stress_corr = np.corrcoef(wind_stress_mean,wind_stress_real_mean)[0,1]
    wind_stress_rmse = np.sqrt(np.mean((wind_stress_mean - wind_stress_real_mean)**2))
    wind_stress_sum_rmse = np.sqrt(np.nanmean((wind_stress - ws_real)**2))
    corr_ws = field_corr(wind_stress,ws_real)
    wind_stress_corr_mean = np.nanmean(corr_ws)   
    # ==============================
    ocean_temp = xa_mean[:,3:]
    ot_real = real_data[:,3:]
    ocean_temp_mean = np.nanmean(ocean_temp,axis=(1,2,3))
    ocean_temp_real_mean = np.nanmean(ot_real,axis=(1,2,3))
    ocean_temp_corr = np.corrcoef(ocean_temp_mean,ocean_temp_real_mean)[0,1]
    ocean_temp_rmse = np.sqrt(np.mean((ocean_temp_mean - ocean_temp_real_mean)**2))
    ocean_temp_sum_rmse = np.sqrt(np.nanmean((ocean_temp - ot_real)**2))
    corr_ot = field_corr(ocean_temp,ot_real)
    ocean_temp_corr_mean = np.nanmean(corr_ot)



    # ==============================


    res =  {'xa_Nino34_corr':xa_Nino34_corr,
            'xa_Nino34_rmse':xa_Nino34_rmse,
            'xp_Nino34_corr':xp_Nino34_corr,
            'xp_Nino34_rmse':xp_Nino34_rmse,
            'sst_mean_corr':sst_mean_corr,
            'sst_mean_rmse':sst_mean_rmse,
            'sst_corr_mean':sst_corr_mean,
            'sst_sum_rmse':sst_sum_rmse,
            'wind_stress_corr':wind_stress_corr,
            'wind_stress_rmse':wind_stress_rmse,
            'wind_stress_sum_rmse':wind_stress_sum_rmse,
            'wind_stress_corr_mean':wind_stress_corr_mean,
            'ocean_temp_corr':ocean_temp_corr,
            'ocean_temp_rmse':ocean_temp_rmse,
            'ocean_temp_sum_rmse':ocean_temp_sum_rmse,
            'ocean_temp_corr_mean':ocean_temp_corr_mean,
            'ecr_p':ecr_p,
            'ecr_a':ecr_a}
    # list(res.values())
    for key in list(res.keys()):
        res[key] = float(res[key])
    return res
        

def save_lite_post(config,res):
    path = config['save_path'] + config['job_name'] + '/' + 'alite_post.txt'
    with open(path,'w') as f:
        f.write(str(res))
    config['logger'].info("Saving Lite Post Process to:" + path)

    # sstm 
    



if __name__ == "__main__":
    # config = {'noise_path':"/glade/work/zilumeng/3D_trans/data/noise1/eof/m0_3x4p_1_leadmonth%s.pkl"}
    # config['seed'] = 0
    # noise_load = load_noise2(config)
    # noise = noise_load.load(pc_num=100,lead_month=1,number=10)
    # print(noise.shape)
    import yaml
    yml_path = "/glade/work/zilumeng/3D_trans/Da/cfg_new.yml"
    with open(yml_path, 'r') as stream:
        config = yaml.safe_load(stream)
    sys.path.append(config['model_config'])
    tmp = config['start_time']

    from myconfig1 import mypara
    config['my_para'] = mypara
    config['start_time'] = dt.datetime(int(tmp[:4]),int(tmp[4:6]),int(tmp[6:8]),int(tmp[8:10]))
    config['current_time'] = dt.datetime(int(tmp[:4]),int(tmp[4:6]),int(tmp[6:8]),int(tmp[8:10]))

    config['Nino34'] = {'xa':[],'xp':[]}
    res = lite_post(config)
    print(res)


