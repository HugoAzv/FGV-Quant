# -*- coding: utf-8 -*-
"""
Created on Fri May 14 09:24:34 2021

@author: lucas
"""
#%% Bibliotecas

import numpy as np
import pandas as pd
import pandas_datareader as web
from pandas_datareader import data
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels as stm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
from statsmodels.tsa.ar_model import AR
from statsmodels.tsa.arima_model import ARMA
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.holtwinters import SimpleExpSmoothing

#%% Dados de Cripto
##### 

## dados históricos
inicio = datetime(2017, 1, 1)
fim = 'today'

'''
btc = data.DataReader('BTC-USD', data_source='yahoo', start= inicio, end='today')

eth = data.DataReader('ETH-USD', data_source='yahoo', start= inicio, end='today')

xrp = data.DataReader('XRP-USD', data_source='yahoo', start= inicio, end='today')

bch = data.DataReader('BCH-USD', data_source='yahoo', start= inicio, end='today')

lik = data.DataReader('LINK-USD', data_source='yahoo', start= inicio, end='today')

dog = data.DataReader('DOGE-USD', data_source='yahoo', start= inicio, end='today')

cad = data.DataReader('ADA-USD', data_source='yahoo', start= inicio, end='today')

eos = data.DataReader('EOS-USD', data_source='yahoo', start= inicio, end='today')

xlm = data.DataReader('XLM-USD', data_source='yahoo', start= inicio, end='today')
'''

##criando array de criptos e um lista
lista_cripto = ['BTC-USD','ETH-USD','XRP-USD','BCH-USD','LINK-USD','DOGE-USD','ADA-USD','BCH-USD','EOS-USD','XLM-USD']
lista_var = ['btc','eth','xrp','bch','lik','dog','cad','bch','eos','xlm']

#%% Base de preços
####  
cript_price=[]

for ticker in lista_cripto:
    phi = data.DataReader(ticker, data_source='yahoo', start= inicio, end= fim)
    phi['Ticket'] = ticker
    cript_price.append(phi)

##Ajustando a tabela
df = pd.concat(cript_price)
df = df.reset_index()
df = df[['Date', 'Close', 'Ticket']]
df.head()

## Arrumando a tabela df em colunas de cada ativo, com linhas de cada data, e
## valores de fechamento. Usando .reset_index para separar a coluna de Data
## e usando de dropna para tirar os valores NaN
df_pivot = df.pivot_table(values='Close',index='Date',columns='Ticket').dropna()
df_pivot.head()

#%% Base de Retornos diários
#### 

df1_pivot = df_pivot.pct_change().dropna()

#%% Estat - Correlação
#### 
### Correlação Cotação
## Criando a tabela de correlação entre os ativos
corr_df = df_pivot.corr(method='pearson')
corr_df.head().reset_index()
corr_df.head(10)

## Criando o mapa termico de Correlação
# Tirar o triângulo superior da tabela, já que se repete
mask = np.zeros_like(corr_df)
mask[np.triu_indices_from(mask)] = True
# Gerar o plot
sns.heatmap(corr_df, cmap='RdYlGn', vmax=1.0, vmin=-1.0 , mask = mask, linewidths=2.5)
plt.yticks(rotation=0) 
plt.xticks(rotation=45) 
plt.show()

#%% #Correlação Retornos(1) 
### 
## Criando a tabela de correlação entre os ativos
corr_df1 = df1_pivot.corr(method='pearson')
corr_df1.head().reset_index()
corr_df1.head(10)

## Criando o mapa termico de Correlação
# Tirar o triângulo superior da tabela, já que se repete
mask1 = np.zeros_like(corr_df1)
mask1[np.triu_indices_from(mask1)] = True
# Gerar o plot
sns.heatmap(corr_df1, cmap='RdYlGn', vmax=1.0, vmin=-1.0 , mask = mask1, linewidths=2.5)
plt.yticks(rotation=0) 
plt.xticks(rotation=45) 
plt.show()

#%% #dev pad - diário
## dev pad - diário
cript_devpad = df1_pivot.std()

#%% >>>Correção de Série Temporal 
#%% Mostrando Tendência, Sazonalidade e Resíduo
### Correção de Série Temporal
## Plotando as colunas
df1_pivot.plot(subplots=True, figsize=(10,8))
df_pivot.plot(subplots=True, figsize=(10,8))

### Caso de Retornos
## Reestruturação dos Dados em Meses e horas
#Meses
df1_pivm = df1_pivot.resample('M').mean()
df1_pivm.plot(subplots=True, figsize=(10,8))
#Horas-O usado no Seasonal_decompose
df1_pivh = df1_pivot.resample('H').ffill()
df1_pivh.plot(subplots=True, figsize=(10,8))


## Mostra de Trend, Sazonalidade e Resíduo
df1_pivh = df1_pivh.rename(columns={'BTC-USD':'BTC'})
resultados = seasonal_decompose(df1_pivh.BTC, model='additive')
resultados.plot()
plt.show()

### Caso de Cotação
## Reestruturação dos Dados em Meses e horas
#Meses
df_pivm = df_pivot.resample('M').mean()
df_pivm.plot(subplots=True, figsize=(10,8))
#Horas-O usado no Seasonal_decompose
df_pivh = df_pivot.resample('H').ffill()
df_pivh.plot(subplots=True, figsize=(10,8))


## Mostra de Trend, Sazonalidade e Resíduo
df_pivh = df_pivh.rename(columns={'BTC-USD':'BTC'})
resultados = seasonal_decompose(df_pivh.BTC, model='additive')
resultados.plot()
plt.show()

#%% Teste Augmented Dickey-Fuller
### Retornos 
adftest1 = adfuller(df1_pivh['BTC'])#Para printar os resultados de maneira elegante:  
out = pd.Series(adftest1[0:4], index=['Teste','p-valor','Lags','Número de observações usadas'])  
for key,value in adftest1[4].items():  
    out['Valor crítico (%s)'%key] = value  
print(out)

### Cotação
adftest = adfuller(df_pivh['BTC'])#Para printar os resultados de maneira elegante:  
out = pd.Series(adftest[0:4], index=['Teste','p-valor','Lags','Número de observações usadas'])  
for key,value in adftest[4].items():  
    out['Valor crítico (%s)'%key] = value  
print(out)

#%% Autocorrelação
#%% White noise + Sinais explicam o modelo
#%% Usando de AUTOREGRESSÃO(AR)
## Gráfico de ACF e PACF
# Retorno
plot_acf(df1_pivh.BTC)
plot_pacf(df1_pivh.BTC)

# Cotação
plot_acf(df_pivh.BTC)
plot_pacf(df_pivh.BTC)
ar = AR(df1_pivh.BTC)  
ar_fit = ar.fit()  
predicted = ar_fit.predict(start=len(df1_pivh.BTC), end=len(df1_pivh.BTC)+7)

#%% Usando de MÉDIA MOVEL(MA)

ma = ARMA(df1_pivh.BTC, order=(0, 3))  
ma_fit = ma.fit()  
predicted = ma_fit.predict(len(df1_pivh.BTC), len(df1_pivh.BTC)+3)

#%% Usando de Autoregressive Integrated Moving Average (ARIMA)

model = ARIMA(df1_pivh.BTC, order=(4, 1, 0))  
#No ARIMA a ordem é para a AR, as diferenciações e a MA, respectivamente.  
model_fit = model.fit()  
predicted = model_fit.predict(len(df1_pivh.BTC), len(df1_pivh.BTC)+3)


