import pandas as pd
from glob import glob

def _get_df(base_url="../public-covid-data", folder="rp_im"):
    pathlist = glob("{}/{}/*".format(base_url,folder))
    filelist = [p.split("/")[-1] for p in pathlist]
    return pd.DataFrame({"FilePath":pathlist,"FileName":filelist})

def get_df_all(base_url="../pubic-covid-data"):
    rp_im_df = _get_df(folder="rp_im")
    rp_msk_df = _get_df(folder="rp_msk")
    return rp_im_df.merge(rp_msk_df,on="FileName",suffixes=("Image","Mask"))    