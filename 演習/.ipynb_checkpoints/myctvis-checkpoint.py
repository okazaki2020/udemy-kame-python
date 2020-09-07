import pandas as pd
from glob import glob
import nibabel as nib
import numpy as np
from matplotlib import pyplot as plt

def _get_df(base_url="../public-covid-data", folder="rp_im"):
    pathlist = glob("{}/{}/*".format(base_url,folder))
    filelist = [p.split("/")[-1] for p in pathlist]
    return pd.DataFrame({"FilePath":pathlist,"FileName":filelist})

def get_df_all(base_url="../pubic-covid-data"):
    rp_im_df = _get_df(folder="rp_im")
    rp_msk_df = _get_df(folder="rp_msk")
    return rp_im_df.merge(rp_msk_df,on="FileName",suffixes=("Image","Mask"))

#NifTIデータをnumpy arrayとしてロードする。
def load_nifti(path):
    nifti = nib.load(path)
    data = nifti.get_fdata()
    data_rolled = np.rollaxis(data, 1)
    return data_rolled

#maskをRGBに変換
def label_color(mask_volume,
                ggo_color=[255,0,0],
                consolidation_color=[0,255,0],
                effusion_color=[0,0,255]):

    shp = mask_volume.shape
    #箱作成
    mask_color = np.zeros((shp[0],shp[1],shp[2],3),dtype=np.float32)
    #色付け
    mask_color[np.equal(mask_volume,1)] = ggo_color
    mask_color[np.equal(mask_volume,2)] = consolidation_color
    mask_color[np.equal(mask_volume,3)] = effusion_color
    
    return mask_color

#CTデータのhuをグレイスケールに変換
def hu_to_gray(volume):
    humax = volume.max()
    humin = volume.min()
    volume_rerange = (volume - humin) / (humax - humin)
    volume_rerange = volume_rerange * 255
    volume_rerange = np.stack([volume_rerange , volume_rerange, volume_rerange],axis=-1)
    
    return volume_rerange.astype(np.uint8)

#オーバーレイ
def overlay(gray_volume, mask_volume, mask_color, alpha=0.3):
    mask_filter = np.greater(mask_volume,0)
    mask_filter = np.stack([mask_filter, mask_filter, mask_filter],axis=-1)
    overlayed = np.where(mask_filter, gray_volume*(1-alpha) + mask_color*alpha, gray_volume).astype(np.uint8) 
    
    return overlayed

#CT画像を並べて表示
def vis_overlay(overlay,
                original_volume,
                mask_volume,
                cols=4,display_num=20,figsize=(20,20)):
    
    rows = (display_num - 1) // cols + 1 
    total_num = overlay.shape[-2]
    fig, ax = plt.subplots(rows,cols,figsize=figsize)
    interval = total_num / display_num
    if interval < 1:
        interval = 1
        
    for i in range(display_num):
        row_i = i//cols
        col_i = i%cols
        idx = int(i * interval)
        #統計値取得
        stats = get_hu_stats(original_volume[:,:,idx],mask_volume[:,:,idx])
        title = 'slice #: {}'.format(idx)
        title += '\nggo mean:{:.0f}±{:.0f}'.format(stats['ggo_mean'], stats['ggo_std'])
        title += '\nconsli mean:{:.0f}±{:.0f}'.format(stats['consolidation_mean'], stats['consolidation_std'])
        title += '\neffusion mean:{:.0f}±{:.0f}'.format(stats['effusion_mean'], stats['effusion_std'])
        #plot
        ax[row_i,col_i].imshow(overlay[:,:,idx])
        #plot情報を表示する
        ax[row_i,col_i].set_title(title)
        ax[row_i,col_i].axis("off")
    fig.tight_layout()
        
        
        
#統計値計算
def get_hu_stats(volume,mask_volume,
                label_dict={1: "ggo",2: "consolidation", 3: "effusion"}):
    result = {}
    
    for label in label_dict.keys():
        prefix = label_dict[label]
        roi_hu = volume[np.equal(mask_volume, label)]
        result[prefix + "_mean"] = np.mean(roi_hu)
        result[prefix + "_std"] = np.mean(roi_hu)
    return result