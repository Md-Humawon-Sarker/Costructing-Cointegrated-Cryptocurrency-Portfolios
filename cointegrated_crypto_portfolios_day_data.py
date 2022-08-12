#!/usr/bin/env python
# coding: utf-8

# In[1]:


from statsmodels.tsa.stattools import adfuller
import pandas as pd
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt


# In[ ]:


#Data Cleaning: Proof Type- POW


# In[2]:


bch = pd.read_csv(r'C:\Users\ASUS\Desktop\daydata\daydata_BCH')


# In[3]:


bch


# In[36]:


bch.rename({'close':'bitcoin'}, axis=1, inplace=True)
bch


# In[3]:


btc = pd.read_csv(r'C:\Users\ASUS\Desktop\daydata\daydata_BTC')


# In[37]:


btc.rename({'close':'bitcoin_cash'}, axis=1, inplace=True)
btc


# In[4]:


bitcoinSV = pd.read_csv(r'C:\Users\ASUS\Desktop\daydata\daydata_BSV(bitcoinSV)')


# In[38]:


bitcoinSV.rename({'close':'bitcoinSV'}, axis=1, inplace=True)
bitcoinSV


# In[17]:


eth = pd.read_csv(r'C:\Users\ASUS\Desktop\daydata\daydata_ETH')
eth


# In[40]:


eth.rename({'close':'eth'}, axis=1, inplace=True)
eth


# In[5]:


ltc = pd.read_csv(r'C:\Users\ASUS\Desktop\daydata\daydata_LTC')


# In[41]:


ltc.rename({'close':'ltc'}, axis=1, inplace=True)
ltc


# In[7]:


minota = pd.read_csv(r'C:\Users\ASUS\Desktop\daydata\daydata_MIOTA')


# In[42]:


minota.rename({'close':'minota'}, axis=1, inplace=True)
minota


# In[6]:


monero = pd.read_csv(r'C:\Users\ASUS\Desktop\daydata\daydata_XMR(monero)')


# In[43]:


monero.rename({'close':'monero'}, axis=1, inplace=True)
monero


# In[8]:


etc = pd.read_csv(r'C:\Users\ASUS\Desktop\daydata\daydata_ETC')


# In[44]:


etc.rename({'close':'etc'}, axis=1, inplace=True)
etc


# In[24]:


#Data Merging


# In[ ]:


#Proof Type-POW


# In[73]:


pow = pd.merge(btc, bch,
                  left_on='datetime', right_on='datetime')


# In[64]:


pow1 = pd.merge(pow, bitcoinSV,
                  left_on='datetime', right_on='datetime')


# In[65]:


pow2 = pd.merge(pow1, eth,
                  left_on='datetime', right_on='datetime')


# In[66]:


pow3 = pd.merge(pow2, ltc,
                  left_on='datetime', right_on='datetime')


# In[67]:


pow4 = pd.merge(pow3, minota,
                  left_on='datetime', right_on='datetime')


# In[68]:


pow5 = pd.merge(pow4, monero,
                  left_on='datetime', right_on='datetime')


# In[70]:


pow6 = pd.merge(pow5, etc,
                  left_on='datetime', right_on='datetime')


# In[77]:


pow6


# In[81]:


pow6.info()


# In[84]:


pow = pow6.loc[:, ['datetime', 'bitcoin', 'bitcoinSV_x', 'bitcoin_cash', 'eth_y', 'ltc_y', 'minota_y', 'monero_y', 'etc']]
pow


# In[86]:


pow.rename({'bitcoinSV_x':'bitcoinSV', 'eth_y':'eth', 'ltc_y':'ltc', 'minota_y':'minota', 'monero_y':'monero'}, axis=1, inplace=True)
pow


# In[88]:


pow = pow.dropna()
pow


# In[90]:


#Descriptive Analysis


# In[91]:


pow.describe()


# In[89]:


pow.to_csv(r'C:\Users\ASUS\Desktop\PoW111.csv', index = None)


# In[ ]:





# In[35]:


##ADF test of BTC


# In[4]:


Bitcoin_daydata = Pow.loc[:, 'bitcoin'].values


# In[30]:


from statsmodels.tsa.stattools import adfuller
import pandas as pd
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')


# In[5]:


result = adfuller(Bitcoin_daydata, autolag='AIC')
print(f'ADF Statistic: {result[0]}')
print(f'n_lags: {result[1]}')
print(f'p-value: {result[1]}')
for datetime, bitcoin in result[4].items():
    print('Critial Values:')
    print(f'   {datetime}, {bitcoin}')


# In[6]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
fig, axes = plt.subplots(figsize=(10,7))
plt.plot(Bitcoin_daydata);
plt.xlabel('Time')
plt.ylabel('Price')
plt.title('Bitcoin')


# In[7]:


BitCash_daydata = Pow.loc[:, 'bitcoin_cash'].values


# In[8]:


result = adfuller(BitCash_daydata, autolag='AIC')
print(f'ADF Statistic: {result[0]}')
print(f'n_lags: {result[1]}')
print(f'p-value: {result[1]}')
for datetime, bitcoin_cash in result[4].items():
    print('Critial Values:')
    print(f'   {datetime}, {bitcoin_cash}')


# In[9]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
fig, axes = plt.subplots(figsize=(10,7))
plt.plot(BitCash_daydata);
plt.xlabel('Time')
plt.ylabel('Price')
plt.title('Bitcoin_Cash')


# In[10]:


BitcoinSV_daydata = Pow.loc[:, 'bitcoinSV'].values


# In[52]:


result = adfuller(BitCash_daydata, autolag='AIC')
print(f'ADF Statistic: {result[0]}')
print(f'n_lags: {result[1]}')
print(f'p-value: {result[1]}')
for datetime, bitcoin_cash in result[4].items():
    print('Critial Values:')
    print(f'   {datetime}, {bitcoin_cash}')


# In[53]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
fig, axes = plt.subplots(figsize=(10,7))
plt.plot(BitcoinSV_daydata);
plt.xlabel('Time')
plt.ylabel('Price')
plt.title('BitcoinSV_daydata')


# In[15]:


daydata_ETH = Pow.loc[:, 'eth'].values


# In[16]:


result = adfuller(daydata_ETH, autolag='AIC')
print(f'ADF Statistic: {result[0]}')
print(f'n_lags: {result[1]}')
print(f'p-value: {result[1]}')
for datetime, eth in result[4].items():
    print('Critial Values:')
    print(f'   {datetime}, {eth}')


# In[17]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
fig, axes = plt.subplots(figsize=(10,7))
plt.plot(daydata_ETH);
plt.xlabel('Time')
plt.ylabel('Price')
plt.title('ETH_daydata')


# In[18]:


LTC_daydata = Pow.loc[:, 'ltc'].values


# In[19]:


result = adfuller(LTC_daydata, autolag='AIC')
print(f'ADF Statistic: {result[0]}')
print(f'n_lags: {result[1]}')
print(f'p-value: {result[1]}')
for datetime, ltc in result[4].items():
    print('Critial Values:')
    print(f'   {datetime}, {ltc}')


# In[21]:


fig, axes = plt.subplots(figsize=(10,7))
plt.plot(LTC_daydata);
plt.xlabel('Time')
plt.ylabel('Price')
plt.title('LTC_daydata')


# In[23]:


Miota_daydata = Pow.loc[:, 'minota'].values


# In[24]:


result = adfuller(Miota_daydata, autolag='AIC')
print(f'ADF Statistic: {result[0]}')
print(f'n_lags: {result[1]}')
print(f'p-value: {result[1]}')
for datetime, minota in result[4].items():
    print('Critial Values:')
    print(f'   {datetime}, {minota}')


# In[25]:


fig, axes = plt.subplots(figsize=(10,7))
plt.plot(Miota_daydata);
plt.xlabel('Time')
plt.ylabel('Price')
plt.title('Miota_Daydata')


# In[26]:


Monero_daydata = Pow.loc[:, 'monero'].values


# In[27]:


result = adfuller(Monero_daydata, autolag='AIC')
print(f'ADF Statistic: {result[0]}')
print(f'n_lags: {result[1]}')
print(f'p-value: {result[1]}')
for datetime, monero in result[4].items():
    print('Critial Values:')
    print(f'   {datetime}, {monero}')


# In[29]:


fig, axes = plt.subplots(figsize=(10,7))
plt.plot(Monero_daydata);
plt.xlabel('Time')
plt.ylabel('Price')
plt.title('Monero_daydata')


# In[33]:


ETC_daydata =  Pow.loc[:, 'etc'].values


# In[34]:


result = adfuller(ETC_daydata, autolag='AIC')
print(f'ADF Statistic: {result[0]}')
print(f'n_lags: {result[1]}')
print(f'p-value: {result[1]}')
for datetime, etc in result[4].items():
    print('Critial Values:')
    print(f'   {datetime}, {etc}')


# In[36]:


fig, axes = plt.subplots(figsize=(10,7))
plt.plot(ETC_daydata);
plt.xlabel('Time')
plt.ylabel('Price')
plt.title('ETC_daydata')


# In[84]:


PoW = pd.merge(Pow, proof_type1, how='outer',
                  left_on='datetime', right_on='datetime' )


# In[85]:


PoW.to_csv(r'C:\Users\ASUS\Desktop\Data set\PoW.csv', index = None)


# In[86]:


##Engle-Granger Test 


# In[3]:


PoW_proof = pd.read_csv(r'C:\Users\ASUS\Desktop\Data set\PoW1.csv')


# In[98]:


PoW_proof


# In[125]:


## Cointegration between Bitcoin & BitcoinCash in 11/19/2018 to 6/6/2020


# In[99]:


df = pd.DataFrame(PoW_proof, columns= ['bitcoin','bitcoin_cash'])
X = df["bitcoin"]
y = df["bitcoin_cash"]
from statsmodels.tsa.stattools import coint
print('Results of Engle-Granger Test:')
egtest = coint(y, X, method='aeg', autolag='AIC')
egoutput = pd.Series(egtest[0:3], index=['Test Statistic','p-value','Critical Values 1%,5%,10%'])
print (egoutput)


# In[ ]:


## 11/19/2018 to 6/6/2020


# In[108]:


## Cointegration between BitcoinSV & Bitcoin in 11/19/2018 to 6/6/2020


# In[100]:


df = pd.DataFrame(PoW_proof, columns= ['bitcoin','bitcoinSV'])
X = df["bitcoin"]
y = df["bitcoinSV"]
from statsmodels.tsa.stattools import coint
print('Results of Engle-Granger Test:')
egtest = coint(y, X, method='aeg', autolag='AIC')
egoutput = pd.Series(egtest[0:3], index=['Test Statistic','p-value','Critical Values 1%,5%,10%'])
print (egoutput)


# In[102]:


## Cointegration between BitcoinSV & Etherium in 11/19/2018 to 6/6/2020


# In[101]:


df = pd.DataFrame(PoW_proof, columns= ['bitcoin','eth'])
X = df["bitcoin"]
y = df["eth"]
from statsmodels.tsa.stattools import coint
print('Results of Engle-Granger Test:')
egtest = coint(y, X, method='aeg', autolag='AIC')
egoutput = pd.Series(egtest[0:3], index=['Test Statistic','p-value','Critical Values 1%,5%,10%'])
print (egoutput)


# In[ ]:


## Cointegration between Bitcoin & ltc in 11/19/2018 to 6/6/2020


# In[104]:


df = pd.DataFrame(PoW_proof, columns= ['bitcoin','ltc'])
X = df["bitcoin"]
y = df["ltc"]
from statsmodels.tsa.stattools import coint
print('Results of Engle-Granger Test:')
egtest = coint(y, X, method='aeg', autolag='AIC')
egoutput = pd.Series(egtest[0:3], index=['Test Statistic','p-value','Critical Values 1%,5%,10%'])
print (egoutput)


# In[109]:


##Cointegration between BitcoinSV & minota in 11/19/2018 to 6/6/2020


# In[105]:


df = pd.DataFrame(PoW_proof, columns= ['bitcoin','minota'])
X = df["bitcoin"]
y = df["minota"]
from statsmodels.tsa.stattools import coint
print('Results of Engle-Granger Test:')
egtest = coint(y, X, method='aeg', autolag='AIC')
egoutput = pd.Series(egtest[0:3], index=['Test Statistic','p-value','Critical Values 1%,5%,10%'])
print (egoutput)


# In[113]:


##Cointegration between BitcoinSV & monero in 11/19/2018 to 6/6/2020


# In[106]:


df = pd.DataFrame(PoW_proof, columns= ['bitcoin','monero'])
X = df["bitcoin"]
y = df["monero"]
from statsmodels.tsa.stattools import coint
print('Results of Engle-Granger Test:')
egtest = coint(y, X, method='aeg', autolag='AIC')
egoutput = pd.Series(egtest[0:3], index=['Test Statistic','p-value','Critical Values 1%,5%,10%'])
print (egoutput)


# In[114]:


##Cointegration between BitcoinSV & etc in 11/19/2018 to 6/6/2020


# In[107]:


df = pd.DataFrame(PoW_proof, columns= ['bitcoin','etc'])
X = df["bitcoin"]
y = df["etc"]
from statsmodels.tsa.stattools import coint
print('Results of Engle-Granger Test:')
egtest = coint(y, X, method='aeg', autolag='AIC')
egoutput = pd.Series(egtest[0:3], index=['Test Statistic','p-value','Critical Values 1%,5%,10%'])
print (egoutput)


# In[ ]:





# In[115]:


## Cointegration test from 6/7/2020 to 12/26/2021


# In[48]:


PoW_proof = pd.read_csv(r'C:\Users\ASUS\Desktop\daydata\PoW2.csv')


# In[49]:


PoW_proof


# In[ ]:


## Cointegration between Bitcoin & BitcoinCash in 


# In[50]:


df = pd.DataFrame(PoW_proof, columns= ['bitcoin','bitcoin_cash'])
X = df["bitcoin"]
y = df["bitcoin_cash"]
from statsmodels.tsa.stattools import coint
print('Results of Engle-Granger Test:')
egtest = coint(y, X, method='aeg', autolag='AIC')
egoutput = pd.Series(egtest[0:3], index=['Test Statistic','p-value','Critical Values 1%,5%,10%'])
print (egoutput)


# In[ ]:


## Cointegration between BitcoinSV & Bitcoin in 6/7/2020 to 12/26/2021


# In[51]:


df = pd.DataFrame(PoW_proof, columns= ['bitcoin','bitcoinSV'])
X = df["bitcoin"]
y = df["bitcoinSV"]
from statsmodels.tsa.stattools import coint
print('Results of Engle-Granger Test:')
egtest = coint(y, X, method='aeg', autolag='AIC')
egoutput = pd.Series(egtest[0:3], index=['Test Statistic','p-value','Critical Values 1%,5%,10%'])
print (egoutput)


# In[ ]:


## Cointegration between BitcoinSV & Etherium in 6/7/2020 to 12/26/2021


# In[43]:


df = pd.DataFrame(PoW_proof, columns= ['bitcoin','eth'])
X = df["bitcoin"]
y = df["eth"]
from statsmodels.tsa.stattools import coint
print('Results of Engle-Granger Test:')
egtest = coint(y, X, method='aeg', autolag='AIC')
egoutput = pd.Series(egtest[0:3], index=['Test Statistic','p-value','Critical Values 1%,5%,10%'])
print (egoutput)


# In[ ]:


## Cointegration between Bitcoin & ltc in 6/7/2020 to 12/26/2021


# In[44]:


df = pd.DataFrame(PoW_proof, columns= ['bitcoin','ltc'])
X = df["bitcoin"]
y = df["ltc"]
from statsmodels.tsa.stattools import coint
print('Results of Engle-Granger Test:')
egtest = coint(y, X, method='aeg', autolag='AIC')
egoutput = pd.Series(egtest[0:3], index=['Test Statistic','p-value','Critical Values 1%,5%,10%'])
print (egoutput)


# In[ ]:


##Cointegration between BitcoinSV & minota in 6/7/2020 to 12/26/2021


# In[45]:


df = pd.DataFrame(PoW_proof, columns= ['bitcoin','minota'])
X = df["bitcoin"]
y = df["minota"]
from statsmodels.tsa.stattools import coint
print('Results of Engle-Granger Test:')
egtest = coint(y, X, method='aeg', autolag='AIC')
egoutput = pd.Series(egtest[0:3], index=['Test Statistic','p-value','Critical Values 1%,5%,10%'])
print (egoutput)


# In[ ]:


##Cointegration between BitcoinSV & monero in 6/7/2020 to 12/26/2021


# In[46]:


df = pd.DataFrame(PoW_proof, columns= ['bitcoin','monero'])
X = df["bitcoin"]
y = df["monero"]
from statsmodels.tsa.stattools import coint
print('Results of Engle-Granger Test:')
egtest = coint(y, X, method='aeg', autolag='AIC')
egoutput = pd.Series(egtest[0:3], index=['Test Statistic','p-value','Critical Values 1%,5%,10%'])
print (egoutput)


# In[ ]:


##Cointegration between BitcoinSV & etc in 6/7/2020 to 12/26/2021


# In[47]:


df = pd.DataFrame(PoW_proof, columns= ['bitcoin','etc'])
X = df["bitcoin"]
y = df["etc"]
from statsmodels.tsa.stattools import coint
print('Results of Engle-Granger Test:')
egtest = coint(y, X, method='aeg', autolag='AIC')
egoutput = pd.Series(egtest[0:3], index=['Test Statistic','p-value','Critical Values 1%,5%,10%'])
print (egoutput)


# In[3]:


PoW_proof1 = pd.read_csv(r'C:\Users\ASUS\Desktop\Data set\daydata_AR')


# In[4]:


PoW_proof2 = pd.read_csv(r'C:\Users\ASUS\Desktop\Data set\PoW.csv')


# In[5]:


PoW_final = pd.merge(PoW_proof2, PoW_proof1, how='outer',
                  left_on='datetime', right_on='datetime' )


# In[6]:


PoW_final


# In[20]:


## AFD & Engle-Granger Test of Arewave


# In[7]:


PoW_final.to_csv(r'C:\Users\ASUS\Desktop\Data set\PoW_final.csv', index = None)


# In[21]:


ar = pd.read_csv(r'C:\Users\ASUS\Desktop\Data set\ar.csv')


# In[22]:


ar


# In[92]:


from statsmodels.tsa.stattools import adfuller
import pandas as pd
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')


# In[23]:


ar1 = ar.loc[:, 'arwave'].values


# In[24]:


result = adfuller(ar1, autolag='AIC')
print(f'ADF Statistic: {result[0]}')
print(f'n_lags: {result[1]}')
print(f'p-value: {result[1]}')
for datetime, arwave in result[4].items():
    print('Critial Values:')
    print(f'   {datetime}, {arwave}')


# In[25]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
fig, axes = plt.subplots(figsize=(10,7))
plt.plot(ar1);
plt.xlabel('Time')
plt.ylabel('Price')
plt.title('Arwave Daily_data')


# In[26]:


df = pd.DataFrame(ar, columns= ['bitcoin','arwave'])
X = df["bitcoin"]
y = df["arwave"]
from statsmodels.tsa.stattools import coint
print('Results of Engle-Granger Test:')
egtest = coint(y, X, method='aeg', autolag='AIC')
egoutput = pd.Series(egtest[0:3], index=['Test Statistic','p-value','Critical Values 1%,5%,10%'])
print (egoutput)


# In[27]:





# In[28]:


## Proof Type - DPoS ------------------------ Daily Data


# In[ ]:





# In[30]:


EOS = pd.read_csv(r'C:\Users\ASUS\Desktop\Data set\daydata_EOS')


# In[31]:


Terra = pd.read_csv(r'C:\Users\ASUS\Desktop\Data set\daydata_LUNA')


# In[32]:


BitTorrent = pd.read_csv(r'C:\Users\ASUS\Desktop\Data set\daydata_BTT(bitTorent)')


# In[34]:


Tron = pd.read_csv(r'C:\Users\ASUS\Desktop\Data set\daydata_TRX(Tron)')


# In[35]:


Tezos = pd.read_csv(r'C:\Users\ASUS\Desktop\Data set\daydata_XTZ(Tezos)')


# In[37]:


Dpos = pd.merge(EOS, Terra, how='outer',
                  left_on='datetime', right_on='datetime' )


# In[38]:


Dpos1 = pd.merge(Dpos, BitTorrent, how='outer',
                  left_on='datetime', right_on='datetime' )


# In[39]:


Dpos2 = pd.merge(Dpos1, Tron, how='outer',
                  left_on='datetime', right_on='datetime' )


# In[41]:


Dpos3 = pd.merge(Dpos2, Tezos, how='outer',
                  left_on='datetime', right_on='datetime' )


# In[42]:


Dpos3


# In[43]:


Dpos3.to_csv(r'C:\Users\ASUS\Desktop\Data set\Dpos3.csv', index = None)


# In[ ]:





# In[ ]:





# In[44]:


##   -----------------------------------------  ADF TEST  --------------------------------------------------------


# In[ ]:





# In[ ]:





# In[10]:


DPoS = pd.read_csv(r'C:\Users\ASUS\Desktop\daydata\Dpos3.csv')


# In[11]:


DPoS


# In[14]:


DPoS.describe()


# In[48]:


EOS = DPoS.loc[:, 'eos'].values


# In[50]:


result = adfuller(EOS, autolag='AIC')
print(f'ADF Statistic: {result[0]}')
print(f'n_lags: {result[1]}')
print(f'p-value: {result[1]}')
for datetime, eos in result[4].items():
    print('Critial Values:')
    print(f'   {datetime}, {eos}')


# In[51]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
fig, axes = plt.subplots(figsize=(10,7))
plt.plot(EOS);
plt.xlabel('Time')
plt.ylabel('Price')
plt.title('EOS Daily_data')


# In[52]:


Terra = DPoS.loc[:, 'terra'].values


# In[53]:


result = adfuller(Terra, autolag='AIC')
print(f'ADF Statistic: {result[0]}')
print(f'n_lags: {result[1]}')
print(f'p-value: {result[1]}')
for datetime, terra in result[4].items():
    print('Critial Values:')
    print(f'   {datetime}, {terra}')


# In[54]:


fig, axes = plt.subplots(figsize=(10,7))
plt.plot(Terra);
plt.xlabel('Time')
plt.ylabel('Price')
plt.title('Terra')


# In[55]:


BitTorrent = DPoS.loc[:, 'bittorrent'].values


# In[56]:


result = adfuller(BitTorrent, autolag='AIC')
print(f'ADF Statistic: {result[0]}')
print(f'n_lags: {result[1]}')
print(f'p-value: {result[1]}')
for datetime, bittorrent in result[4].items():
    print('Critial Values:')
    print(f'   {datetime}, {bittorrent}')


# In[66]:


fig, axes = plt.subplots(figsize=(10,7))
plt.plot(BitTorrent);
plt.xlabel('Time')
plt.ylabel('Price')
plt.title('BitTorrent')


# In[58]:


Tron = DPoS.loc[:, 'tron'].values


# In[59]:


result = adfuller(Tron, autolag='AIC')
print(f'ADF Statistic: {result[0]}')
print(f'n_lags: {result[1]}')
print(f'p-value: {result[1]}')
for datetime, tron in result[4].items():
    print('Critial Values:')
    print(f'   {datetime}, {tron}')


# In[60]:


fig, axes = plt.subplots(figsize=(10,7))
plt.plot(Tron);
plt.xlabel('Time')
plt.ylabel('Price')
plt.title('Terra')


# In[61]:


Tezos = DPoS.loc[:, 'tezos'].values


# In[62]:


result = adfuller(Tezos, autolag='AIC')
print(f'ADF Statistic: {result[0]}')
print(f'n_lags: {result[1]}')
print(f'p-value: {result[1]}')
for datetime, tezos in result[4].items():
    print('Critial Values:')
    print(f'   {datetime}, {tezos}')


# In[64]:


fig, axes = plt.subplots(figsize=(10,7))
plt.plot(Tezos);
plt.xlabel('Time')
plt.ylabel('Price')
plt.title('Tezos')


# In[ ]:





# In[65]:


##       -------------------------------------------- Engle-Granger Test -----------------------------------------------


# In[ ]:





# In[5]:


dpos1 = pd.read_csv(r'C:\Users\ASUS\Desktop\daydata\dpos2.csv')


# In[71]:


dpos1


# In[ ]:


## Engle-Granger test of terra and bittorrent from 2/3/2019 to 7/20/2020


# In[75]:


df = pd.DataFrame(dpos1, columns= ['terra','bittorrent'])
X = df["terra"]
y = df["bittorrent"]
from statsmodels.tsa.stattools import coint
print('Results of Engle-Granger Test:')
egtest = coint(y, X, method='aeg', autolag='AIC')
egoutput = pd.Series(egtest[0:3], index=['Test Statistic','p-value','Critical Values 1%,5%,10%'])
print (egoutput)


# In[76]:


## Engle-Granger test of terra and tron from 2/3/2019 to 7/20/2020


# In[77]:


df = pd.DataFrame(dpos1, columns= ['terra','tron'])
X = df["terra"]
y = df["tron"]
from statsmodels.tsa.stattools import coint
print('Results of Engle-Granger Test:')
egtest = coint(y, X, method='aeg', autolag='AIC')
egoutput = pd.Series(egtest[0:3], index=['Test Statistic','p-value','Critical Values 1%,5%,10%'])
print (egoutput)


# In[78]:


## Engle-Granger test of terra and tezos from 2/3/2019 to 7/20/2020


# In[79]:


df = pd.DataFrame(dpos1, columns= ['terra','tezos'])
X = df["terra"]
y = df["tezos"]
from statsmodels.tsa.stattools import coint
print('Results of Engle-Granger Test:')
egtest = coint(y, X, method='aeg', autolag='AIC')
egoutput = pd.Series(egtest[0:3], index=['Test Statistic','p-value','Critical Values 1%,5%,10%'])
print (egoutput)


# In[ ]:





# In[82]:


##               ---------------------------- Eangle-Granger Test --------------------------------


# In[ ]:





# In[83]:


dpos1 = pd.read_csv(r'C:\Users\ASUS\Desktop\Data set\dpos1.csv')


# In[85]:


dpos1


# In[86]:


## Engle-Granger test of terra and bittorrent from 7/20/2020 to 12/26/2021


# In[87]:


df = pd.DataFrame(dpos1, columns= ['terra','bittorrent'])
X = df["terra"]
y = df["bittorrent"]
from statsmodels.tsa.stattools import coint
print('Results of Engle-Granger Test:')
egtest = coint(y, X, method='aeg', autolag='AIC')
egoutput = pd.Series(egtest[0:3], index=['Test Statistic','p-value','Critical Values 1%,5%,10%'])
print (egoutput)


# In[ ]:





# In[89]:


df = pd.DataFrame(dpos1, columns= ['terra','tron'])
X = df["terra"]
y = df["tron"]
from statsmodels.tsa.stattools import coint
print('Results of Engle-Granger Test:')
egtest = coint(y, X, method='aeg', autolag='AIC')
egoutput = pd.Series(egtest[0:3], index=['Test Statistic','p-value','Critical Values 1%,5%,10%'])
print (egoutput)


# In[ ]:





# In[90]:


df = pd.DataFrame(dpos1, columns= ['terra','tezos'])
X = df["terra"]
y = df["tezos"]
from statsmodels.tsa.stattools import coint
print('Results of Engle-Granger Test:')
egtest = coint(y, X, method='aeg', autolag='AIC')
egoutput = pd.Series(egtest[0:3], index=['Test Statistic','p-value','Critical Values 1%,5%,10%'])
print (egoutput)


# In[ ]:





# In[1]:


##                      -----------------Proof Type: POS----------------------


# In[ ]:





# In[1]:


import pandas as pd


# In[5]:


cardano = pd.read_csv(r'C:\Users\ASUS\Desktop\Data set\daydata_ADA(cardano)')


# In[38]:


algorand = pd.read_csv(r'C:\Users\ASUS\Desktop\Data set\daydata_ALGO')


# In[39]:


algorand


# In[7]:


anchor_protocol = pd.read_csv(r'C:\Users\ASUS\Desktop\Data set\daydata_ANC')


# In[8]:


cosmos = pd.read_csv(r'C:\Users\ASUS\Desktop\Data set\daydata_ATOM')


# In[9]:


avalanche = pd.read_csv(r'C:\Users\ASUS\Desktop\Data set\daydata_AVAX')


# In[13]:


crypto_com = pd.read_csv(r'C:\Users\ASUS\Desktop\Data set\daydata_CRO(crypto.com)')


# In[14]:


hedera_hashgraph = pd.read_csv(r'C:\Users\ASUS\Desktop\Data set\daydata_HBAR(hashbar)')


# In[16]:


internet_computer = pd.read_csv(r'C:\Users\ASUS\Desktop\Data set\daydata_ICP(internet computer)')


# In[17]:


near = pd.read_csv(r'C:\Users\ASUS\Desktop\Data set\daydata_NEAR')


# In[21]:


PayProtocol = pd.read_csv(r'C:\Users\ASUS\Desktop\Data set\daydata_PCI(Paycoin Price )')


# In[22]:


Oasis_labs = pd.read_csv(r'C:\Users\ASUS\Desktop\Data set\daydata_ROSE')


# In[23]:


terraUSD = pd.read_csv(r'C:\Users\ASUS\Desktop\Data set\daydata_UST(terraUSD)')


# In[ ]:





# In[35]:


POS10 = pd.merge(POS9, terraUSD, how='outer',
                  left_on='datetime', right_on='datetime' )


# In[36]:


POS10


# In[37]:


POS10.to_csv(r'C:\Users\ASUS\Desktop\Data set\POS.csv', index = None)


# In[ ]:





# In[40]:


##                          ----------------POS1----ADF TEST----------------


# In[2]:


POS = pd.read_csv(r'C:\Users\ASUS\Desktop\daydata\POS.csv')
POS


# In[29]:


POS.describe()


# In[ ]:





# In[3]:


Pos1 = POS.loc[:, ['datetime', 'cardano', 'algorand', 'cosmos', 'crypto_com', 'hedera_hashgraph',]]
Pos1


# In[4]:


Pos1 = Pos1.dropna()


# In[46]:


##   -------------------------------AFD TEST: pos1------------------------------------


# In[21]:


Pos1


# In[5]:


cardano_daydata = Pos1.loc[:, 'cardano'].values


# In[6]:


result = adfuller(cardano_daydata, autolag='AIC')
print(f'ADF Statistic: {result[0]}')
print(f'n_lags: {result[1]}')
print(f'p-value: {result[1]}')
for datetime, cardano in result[4].items():
    print('Critial Values:')
    print(f'   {datetime}, {cardano}')


# In[7]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
fig, axes = plt.subplots(figsize=(10,7))
plt.plot(cardano_daydata);
plt.xlabel('Time')
plt.ylabel('Price')
plt.title('Cardano')


# In[ ]:





# In[8]:


algorand_daydata =   Pos1.loc[:, 'algorand'].values


# In[9]:


result = adfuller(algorand_daydata, autolag='AIC')
print(f'ADF Statistic: {result[0]}')
print(f'n_lags: {result[1]}')
print(f'p-value: {result[1]}')
for datetime, algorand in result[4].items():
    print('Critial Values:')
    print(f'   {datetime}, {algorand}')
    


# In[10]:


fig, axes = plt.subplots(figsize=(10,7))
plt.plot(algorand_daydata);
plt.xlabel('Time')
plt.ylabel('Price')
plt.title('algorand_daydata')


# In[ ]:





# In[11]:


cosmos_daydata = Pos1.loc[:, 'cosmos'].values


# In[12]:


result = adfuller(cosmos_daydata, autolag='AIC')
print(f'ADF Statistic: {result[0]}')
print(f'n_lags: {result[1]}')
print(f'p-value: {result[1]}')
for datetime, cosmos in result[4].items():
    print('Critial Values:')
    print(f'   {datetime}, {cosmos}')


# In[13]:


fig, axes = plt.subplots(figsize=(10,7))
plt.plot(cosmos_daydata);
plt.xlabel('Time')
plt.ylabel('Price')
plt.title('cosmos_daydata')


# In[14]:


crypto_com_daydata = Pos1.loc[:, 'crypto_com'].values


# In[15]:


result = adfuller(crypto_com_daydata, autolag='AIC')
print(f'ADF Statistic: {result[0]}')
print(f'n_lags: {result[1]}')
print(f'p-value: {result[1]}')
for datetime, crypto_com in result[4].items():
    print('Critial Values:')
    print(f'   {datetime}, {crypto_com}')


# In[16]:


fig, axes = plt.subplots(figsize=(10,7))
plt.plot(crypto_com_daydata);
plt.xlabel('Time')
plt.ylabel('Price')
plt.title('crypto_com')


# In[ ]:





# In[17]:


hedera_hashgraph_daydata = Pos1.loc[:, 'hedera_hashgraph'].values


# In[18]:


result = adfuller(hedera_hashgraph_daydata, autolag='AIC')
print(f'ADF Statistic: {result[0]}')
print(f'n_lags: {result[1]}')
print(f'p-value: {result[1]}')
for datetime, hedera_hashgraph in result[4].items():
    print('Critial Values:')
    print(f'   {datetime}, {hedera_hashgraph}')


# In[19]:


fig, axes = plt.subplots(figsize=(10,7))
plt.plot(hedera_hashgraph_daydata);
plt.xlabel('Time')
plt.ylabel('Price')
plt.title('hedera_hashgraph_daydata')


# In[ ]:





# In[130]:


##             ---------------Engle-Granger Test Pos1--------------------- from 9/20/2019 to 11/5/2020


# In[ ]:





# In[20]:


Pos1


# In[22]:


df = pd.DataFrame(Pos1, columns= ['cardano','algorand'])
X = df["cardano"]
y = df["algorand"]
from statsmodels.tsa.stattools import coint
print('Results of Engle-Granger Test:')
egtest = coint(y, X, method='aeg', autolag='AIC')
egoutput = pd.Series(egtest[0:3], index=['Test Statistic','p-value','Critical Values 1%,5%,10%'])
print (egoutput)


# In[ ]:





# In[23]:


df = pd.DataFrame(Pos1, columns= ['cardano','cosmos'])
X = df["cardano"]
y = df["cosmos"]
from statsmodels.tsa.stattools import coint
print('Results of Engle-Granger Test:')
egtest = coint(y, X, method='aeg', autolag='AIC')
egoutput = pd.Series(egtest[0:3], index=['Test Statistic','p-value','Critical Values 1%,5%,10%'])
print (egoutput)


# In[ ]:





# In[24]:


df = pd.DataFrame(Pos1, columns= ['cardano','crypto_com'])
X = df["cardano"]
y = df["crypto_com"]
from statsmodels.tsa.stattools import coint
print('Results of Engle-Granger Test:')
egtest = coint(y, X, method='aeg', autolag='AIC')
egoutput = pd.Series(egtest[0:3], index=['Test Statistic','p-value','Critical Values 1%,5%,10%'])
print (egoutput)


# In[ ]:





# In[25]:


df = pd.DataFrame(Pos1, columns= ['cardano','hedera_hashgraph'])
X = df["cardano"]
y = df["hedera_hashgraph"]
from statsmodels.tsa.stattools import coint
print('Results of Engle-Granger Test:')
egtest = coint(y, X, method='aeg', autolag='AIC')
egoutput = pd.Series(egtest[0:3], index=['Test Statistic','p-value','Critical Values 1%,5%,10%'])
print (egoutput)


# In[ ]:





# In[146]:


## ----------------------------ADF Test: POS2-----------------------------------near	PayProtocol	Oasis_labs	terraUSD


# In[27]:


Pos2 = POS.loc[:, ['datetime', 'near', 'PayProtocol', 'Oasis_labs', 'terraUSD']]
Pos2


# In[38]:


Pos2 = Pos2.dropna()
Pos2


# In[ ]:





# In[30]:


near_daydata = Pos2.loc[:, 'near'].values


# In[31]:


result = adfuller(near_daydata, autolag='AIC')
print(f'ADF Statistic: {result[0]}')
print(f'n_lags: {result[1]}')
print(f'p-value: {result[1]}')
for datetime, near in result[4].items():
    print('Critial Values:')
    print(f'   {datetime}, {near}')


# In[32]:


fig, axes = plt.subplots(figsize=(10,7))
plt.plot(near_daydata);
plt.xlabel('Time')
plt.ylabel('Price')
plt.title('near_daydata')


# In[ ]:





# In[35]:


PayProtocol_daydata = Pos2.loc[:, 'PayProtocol'].values


# In[36]:


result = adfuller(PayProtocol_daydata, autolag='AIC')
print(f'ADF Statistic: {result[0]}')
print(f'n_lags: {result[1]}')
print(f'p-value: {result[1]}')
for datetime, PayProtocol in result[4].items():
    print('Critial Values:')
    print(f'   {datetime}, {PayProtocol}')


# In[37]:


fig, axes = plt.subplots(figsize=(10,7))
plt.plot(PayProtocol_daydata);
plt.xlabel('Time')
plt.ylabel('Price')
plt.title('PayProtocol_daydata')


# In[ ]:





# In[40]:


Oasis_labs_daydata = Pos2.loc[:, 'Oasis_labs'].values


# In[41]:


result = adfuller(Oasis_labs_daydata, autolag='AIC')
print(f'ADF Statistic: {result[0]}')
print(f'n_lags: {result[1]}')
print(f'p-value: {result[1]}')
for datetime, Oasis_labs in result[4].items():
    print('Critial Values:')
    print(f'   {datetime}, {Oasis_labs}')


# In[42]:


fig, axes = plt.subplots(figsize=(10,7))
plt.plot(Oasis_labs_daydata);
plt.xlabel('Time')
plt.ylabel('Price')
plt.title('Oasis_labs_daydata')


# In[ ]:





# In[192]:


terraUSD_daydata = Pos2.loc[:, 'terraUSD'].values


# In[193]:


result = adfuller(terraUSD_daydata, autolag='AIC')
print(f'ADF Statistic: {result[0]}')
print(f'n_lags: {result[1]}')
print(f'p-value: {result[1]}')
for datetime, terraUSD in result[4].items():
    print('Critial Values:')
    print(f'   {datetime}, {terraUSD}')


# In[216]:


fig, axes = plt.subplots(figsize=(10,7))
plt.plot(terraUSD_daydata);
plt.xlabel('Time')
plt.ylabel('Price')
plt.title('terraUSD')


# In[ ]:





# In[172]:


##-------------------------Engle-Granger Test-------------------------------------


# In[195]:


df = pd.DataFrame(Pos2, columns= ['near','PayProtocol'])
X = df["near"]
y = df["PayProtocol"]
from statsmodels.tsa.stattools import coint
print('Results of Engle-Granger Test:')
egtest = coint(y, X, method='aeg', autolag='AIC')
egoutput = pd.Series(egtest[0:3], index=['Test Statistic','p-value','Critical Values 1%,5%,10%'])
print (egoutput)


# In[ ]:





# In[196]:


df = pd.DataFrame(Pos2, columns= ['near','Oasis_labs'])
X = df["near"]
y = df["Oasis_labs"]
from statsmodels.tsa.stattools import coint
print('Results of Engle-Granger Test:')
egtest = coint(y, X, method='aeg', autolag='AIC')
egoutput = pd.Series(egtest[0:3], index=['Test Statistic','p-value','Critical Values 1%,5%,10%'])
print (egoutput)


# In[ ]:





# In[197]:


df = pd.DataFrame(Pos2, columns= ['near','terraUSD'])
X = df["near"]
y = df["terraUSD"]
from statsmodels.tsa.stattools import coint
print('Results of Engle-Granger Test:')
egtest = coint(y, X, method='aeg', autolag='AIC')
egoutput = pd.Series(egtest[0:3], index=['Test Statistic','p-value','Critical Values 1%,5%,10%'])
print (egoutput)


# In[ ]:





# In[201]:


##       ---------------------ADF Test----------------POS3----internet_computer	avalanche	anchor_protocol


# In[63]:


Pos3 = POS.loc[:, ['datetime', 'internet_computer', 'avalanche', 'anchor_protocol']]
Pos3


# In[65]:


Pos3 = Pos3.dropna()
Pos3


# In[8]:


internet_computer_daydata = Pos3.loc[:, 'internet_computer'].values


# In[9]:


result = adfuller(internet_computer_daydata, autolag='AIC')
print(f'ADF Statistic: {result[0]}')
print(f'n_lags: {result[1]}')
print(f'p-value: {result[1]}')
for datetime,internet_computer in result[4].items():
    print('Critial Values:')
    print(f'   {datetime}, {internet_computer}')


# In[10]:


import matplotlib.pyplot as plt


# In[11]:


fig, axes = plt.subplots(figsize=(10,7))
plt.plot(internet_computer_daydata);
plt.xlabel('Time')
plt.ylabel('Price')
plt.title('internet_computer')


# In[12]:


avalanche_daydata = Pos3.loc[:, 'avalanche'].values


# In[13]:


result = adfuller(avalanche_daydata, autolag='AIC')
print(f'ADF Statistic: {result[0]}')
print(f'n_lags: {result[1]}')
print(f'p-value: {result[1]}')
for datetime,avalanche in result[4].items():
    print('Critial Values:')
    print(f'   {datetime}, {avalanche}')


# In[14]:


fig, axes = plt.subplots(figsize=(10,7))
plt.plot(avalanche_daydata);
plt.xlabel('Time')
plt.ylabel('Price')
plt.title('avalanche_daydata')


# In[15]:


anchor_protocol_daydata = Pos3.loc[:, 'anchor_protocol'].values


# In[16]:


result = adfuller(anchor_protocol_daydata, autolag='AIC')
print(f'ADF Statistic: {result[0]}')
print(f'n_lags: {result[1]}')
print(f'p-value: {result[1]}')
for datetime,anchor_protocol in result[4].items():
    print('Critial Values:')
    print(f'   {datetime}, {anchor_protocol}')


# In[17]:


fig, axes = plt.subplots(figsize=(10,7))
plt.plot(anchor_protocol_daydata);
plt.xlabel('Time')
plt.ylabel('Price')
plt.title('anchor_protocol')


# In[92]:


# Cointegration among these three cryptocurriences are not possible because internet_compute and anchor_protocol show 
#stationary behavior although p value of avalanche show it is non-stationary.


# In[ ]:




