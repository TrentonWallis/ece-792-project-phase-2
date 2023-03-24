from re import I
import sewar.full_ref as metric
from helper_functions import *
import time
''' pip install sewar

about metrics: https://github.com/andrewekhalel/sewar
 Mean Squared Error (MSE)
 Root Mean Squared Error (RMSE)
 Peak Signal-to-Noise Ratio (PSNR) [1]
 Structural Similarity Index (SSIM) [1]
 Universal Quality Image Index (UQI) [2]
 Multi-scale Structural Similarity Index (MS-SSIM) [3]
 Erreur Relative Globale Adimensionnelle de Synthèse (ERGAS) [4]
 Spatial Correlation Coefficient (SCC) [5]
 Relative Average Spectral Error (RASE) [6]
 Spectral Angle Mapper (SAM) [7]
 Spectral Distortion Index (D_lambda) [8]
 Spatial Distortion Index (D_S) [8]
 Quality with No Reference (QNR) [8]
 Visual Information Fidelity (VIF) [9]
 Block Sensitive - Peak Signal-to-Noise Ratio (PSNR-B) [10]
'''

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as func
import torch.optim as optim
from collections import Counter

def ssim_avg(img1,img2):
  c1 = metric.ssim(img1[0],img2[0])[0]
  c2 = metric.ssim(img1[1],img2[1])[0]
  c3 = metric.ssim(img1[2],img2[2])[0]
  return (c1+c2+c3)/3

def calculate_metrics(img1,img2):
    '''https://sewar.readthedocs.io/en/latest/'''
    
    metrics = {
        #metric.ergas.__name__ : metric.ergas(img1,img2),
        #metric.mse.__name__ : metric.mse(img1,img2),
        #metric.msssim.__name__ : metric.msssim(img1,img2),
        metric.psnr.__name__ : metric.psnr(img1,img2),
        #metric.psnrb.__name__ : metric.psnrb(img1,img2),
        #metric.rase.__name__ : metric.rase(img1,img2),
        #metric.rmse.__name__ : metric.rmse(img1,img2),
        #metric.rmse_sw.__name__ : metric.rmse_sw(img1,img2),
        #metric.sam.__name__ : metric.sam(img1,img2),
        #metric.scc.__name__ : metric.scc(img1,img2),
        metric.ssim.__name__ : ssim_avg(img1,img2),
        #metric.uqi.__name__ : metric.uqi(img1,img2),
        #metric.vifp.__name__ : metric.vifp(img1,img2),
    }
    return metrics

class Benchmark_Suite():
  def __init__(self):
    self.original = None
    self.output = None
    return

  def forward_metrics(self):
    if type(self.original)==None or type(self.output)==None:
      print('Inputs and Output Images not loaded into suite.')
      return
    results = []
    for img1,img2 in zip(self.original,self.output):
        results.append(calculate_metrics(img1,img2))
    return results
  

  def dataset_performance(self,network,loss_func,data_loader,stop=None,per_image=False, average = False):
    losses = []
    metrics = []
    times = []
    network.eval()
    with torch.no_grad():
      for idx, batch in enumerate(data_loader):
        start_time = time.perf_counter()
        output = network(batch["Compressed"])
        times.append(time.perf_counter() - start_time)
        original = batch["Original"]
        losses.append(loss_func(output,original))

        original= 255* original.numpy()
        output= 255* output.numpy()
        self.original = original.astype(np.uint8)
        self.output = output.astype(np.uint8)
        metrics.append(self.forward_metrics())
        if stop == idx:
          break
    # Convert losses to np array 
    losses = np.array(losses)
    times = np.array(times)

    if per_image==False:
      # Average between every image in the batch 
      temp = []
      for batch in metrics:
        total = sum(map(Counter, batch), Counter())
        N = float(len(batch))
        temp.append({ k: v/N for k, v in total.items() })
      metrics = temp
      print(metrics)
      if average: # Averages between each batch
        average_metrics = [{}]
        for key in metrics[0]:
          s = 0
          for metric in metrics:
            s += metric[key]
          average_metrics[0][key] = s/len(metrics)
        metrics = average_metrics
        times = times.mean()

    return losses, metrics, times

