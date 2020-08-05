#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 19:16:00 2019

@author: Pep
"""
#%% imports
import numpy as np
import matplotlib.pyplot as plt  # To visualize
import pandas as pd  # To read data
from scipy.stats import gumbel_r

plt.style.use('ggplot')


#%% read data
raw_data = pd.read_csv('prec.txt', sep ='\t', decimal = ",", index_col =['Year'],usecols=['Year','Pmax'])

#%% filter data
flt_data = raw_data[(raw_data['Pmax']!="nd")]
flt_data.dropna()
c_data = flt_data.apply(lambda x: x.str.replace(',','.'))
c_data = c_data.astype(float)
stts = c_data.describe()
mean = stts['Pmax']['mean']
std = stts['Pmax']['std']
#%% find alpha and beta.
beta = std*np.sqrt(6)/np.pi
alpha = mean - 0.57721*beta
x = np.arange(10,220,1)
df = gumbel_r.pdf(x,alpha,beta)

#%% plotting
plt.figure(figsize=(30,10))

FD = int((stts['Pmax']['max'] - stts['Pmax']['min'])/(2*(stts['Pmax']['75%']-stts['Pmax']['25%'])*stts['Pmax']['count']**(-1/3)))

ax1 = c_data.plot(kind = 'hist', density = True, bins=FD, label='Dades')
ax2 = plt.plot(x,df,label='Distribució de Gumbel')
plt.legend()
plt.xlabel('Precipitació màxima anual [mm]')
plt.ylabel('Freqüència')
plt.title('Precipitacions màximes anuals a Girona i Funció de distribució Gumbel', fontsize = 10)
plt.savefig('distr.eps')
plt.show()
plt.clf()

#%% i'm scum
T = np.array([500, 100, 25, 10])
df_val_meet = [1-1/i for i in T]
pr = -np.log(-np.log(df_val_meet))*beta + alpha
np.savetxt('prec_prd_rtrn.txt', list(zip(pr, T)), fmt="%1f", header = "Pmax \t Periode de Retorn")
