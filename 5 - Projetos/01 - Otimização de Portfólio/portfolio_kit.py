import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


from selenium import webdriver
from pandas_datareader import data


sns.set()


def baixar_preços(tickers, inicio, fim):
    """
    Função que irá buscar os preços de fechamento no Yahoo Finance e retornar esses valores em um DataFrame
    
    tickers = lista com todos os ativos para baixar as informações de preço de fechamento sem terminlogia de ".SA"
    inicio = data de início do conjunto de dados para baixar os dados
    fim = data de fim do conjunto de dados para baixar os dados
    """
    # DataFrame para carregar os preços dos tickers
    prices = pd.DataFrame()
    # Baixar preços no Yahoo
    for row_number, ticker in enumerate(tickers):
        if tickers[row_number][0] != "^":
            prices[ticker] = data.DataReader(ticker+'.SA', data_source='yahoo', start=inicio, end=fim)['Adj Close']
        else:
            prices[ticker] = data.DataReader(ticker, data_source='yahoo', start=inicio, end=fim)['Adj Close']
    # Ajustar o Índice e a coluna do IBOV
    prices.index.name = 'Date'
    return prices



def salvar_preços(prices, nome_arquivo):
    """
    Salva o dataframe em questão no formato csv
    
    prices = dataframe para ser salvo como csv
    """
    # Salvar o DataFrame para futuras aplicações
    if nome_arquivo[-4:] == '.csv':
        prices.to_csv(nome_arquivo)
    else:
        prices.to_csv(nome_arquivo+'.csv')



def calc_retorno(df, drop_na=True):
    """
    Calcula o retorno do DataFrame
    
    df = Dataframe com preços de fechamento
    drop_na = variável para dropar todos os NaN ou não
    """
    if drop_na:
        rets = df.pct_change().dropna()
    else:
        rets = df.pct_change()
    return rets



def annualize_rets(r, periods_per_year=12):
    """
    Anualiza um conjunto de retornos
    
    r = retornos, lido como pd.Series
    periods_per_year = 12
    """
    compounded_growth = (1+r).prod()
    n_periods = r.shape[0]
    return compounded_growth**(periods_per_year/n_periods)-1
            


def annualize_vol(r, periods_per_year):
    """
    Annualizes the vol of  a set of returns
    
    r = pd.Series
    periods_per_year = 12
    """
    return r.std()*(periods_per_year**0.5)


def sharpe_ratio(r, riskfree_rate, periods_per_year=12):
    """
    Computa o valor anualizado do Sharpe Ratio de algum conjunto de retornos
    
    r = retornos dos ativos
    riskfree_rate = taxa livre de risco
    periods_per_year = periodos no ano   
    """
    # convert the annual riskfree rate to per period
    rf_per_period = (1+riskfree_rate)**(1/periods_per_year)-1
    excess_ret = r - rf_per_period
    ann_ex_ret = annualize_rets(excess_ret, periods_per_year)
    ann_vol = annualize_vol(r, periods_per_year)
    return ann_ex_ret/ann_vol


def backtest_peso_fixo(r, weighting, estimation_window=60):
    """
    Backtests a given weighting scheme, given some parameters:
    r : asset returns to use to build the portfolio
    estimation_window: the window to use to estimate parameters
    weighting: the weighting scheme to use, must be a function that takes "r", and a variable number of keyword-value arguments
    """
    n_periods = r.shape[0]
    windows = [(start, start+estimation_window) for start in range(n_periods-estimation_window+1)]
    # windows is a list of tuples which gives us the (integer) location of the start and stop (non inclusive)
    # for each estimation window
    weights = [weighting for win in windows]
    # List -> DataFrame
    weights = pd.DataFrame(weights, index=r.iloc[estimation_window-1:].index, columns=r.columns)
    # return weights
    returns = (weights * r).sum(axis="columns",  min_count=1) #mincount is to generate NAs if all inputs are NAs
    return returns


def summary_stats(r, riskfree_rate=0.03):
    """
    Return a DataFrame that contains aggregated summary stats for the returns in the columns of r
    """
    ann_r = r.aggregate(annualize_rets, periods_per_year=12)
    ann_vol = r.aggregate(annualize_vol, periods_per_year=12)
    ann_sr = r.aggregate(sharpe_ratio, riskfree_rate=riskfree_rate, periods_per_year=12)
    dd = r.aggregate(lambda r: drawdown(r).Drawdown.min())
    skew = r.aggregate(skewness)
    kurt = r.aggregate(kurtosis)
    cf_var5 = r.aggregate(var_gaussian, modified=True)
    hist_cvar5 = r.aggregate(cvar_historic)
    return pd.DataFrame({
        "Annualized Return": ann_r,
        "Annualized Vol": ann_vol,
        "Skewness": skew,
        "Kurtosis": kurt,
        "Cornish-Fisher VaR (5%)": cf_var5,
        "Historic CVaR (5%)": hist_cvar5,
        "Sharpe Ratio": ann_sr,
        "Max Drawdown": dd
    })