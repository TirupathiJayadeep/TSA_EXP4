# Ex.No:04   FIT ARMA MODEL FOR TIME SERIES
# Date: 1-08-2026
### AIM:
To implement ARMA model in python.
### ALGORITHM:
1. Import necessary libraries.
2. Set up matplotlib settings for figure size.
3. Define an ARMA(1,1) process with coefficients ar1 and ma1, and generate a sample of 1000

data points using the ArmaProcess class. Plot the generated time series and set the title and x-
axis limits.

4. Display the autocorrelation and partial autocorrelation plots for the ARMA(1,1) process using
plot_acf and plot_pacf.
5. Define an ARMA(2,2) process with coefficients ar2 and ma2, and generate a sample of 10000

data points using the ArmaProcess class. Plot the generated time series and set the title and x-
axis limits.

6. Display the autocorrelation and partial autocorrelation plots for the ARMA(2,2) process using
plot_acf and plot_pacf.

### PROGRAM:
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# =====================================================
# Load Dataset
# =====================================================
data = pd.read_csv("Month_Value_1.csv")

# Convert Period to datetime
data["Period"] = pd.to_datetime(data["Period"], format="%d.%m.%Y")

# Remove missing Revenue values
data = data.dropna(subset=["Revenue"])

# Sort by date
data = data.sort_values("Period")

# Reset index
data = data.reset_index(drop=True)

# Time Series
X = data["Revenue"]

print("Rows:", len(X))
print("Missing values:", X.isnull().sum())

# =====================================================
# Original Revenue Plot
# =====================================================
plt.figure(figsize=(12,5))
plt.plot(X.values)
plt.title("Original Revenue Data")
plt.xlabel("Observation")
plt.ylabel("Revenue")
plt.xlim(0, len(X)-1)
plt.show()

# =====================================================
# Original Revenue ACF & PACF
# =====================================================
fig, ax = plt.subplots(2,1,figsize=(12,8))

plot_acf(X,
         lags=20,
         ax=ax[0],
         zero=False)

ax[0].set_title("Revenue ACF")
ax[0].set_xlim(0,20)

plot_pacf(X,
          lags=20,
          ax=ax[1],
          method="ywm",
          zero=False)

ax[1].set_title("Revenue PACF")
ax[1].set_xlim(0,20)

plt.tight_layout()
plt.show()

# =====================================================
# ARMA(1,1)
# =====================================================
arma11_model = ARIMA(X, order=(1,0,1)).fit()

print("\nARMA(1,1) Parameters")
print(arma11_model.params)

phi1 = arma11_model.params["ar.L1"]
theta1 = arma11_model.params["ma.L1"]

ar1 = np.array([1,-phi1])
ma1 = np.array([1,theta1])

N = 1000

arma11 = ArmaProcess(ar1,ma1)
ARMA1 = arma11.generate_sample(nsample=N)

plt.figure(figsize=(12,5))
plt.plot(ARMA1)
plt.title("Simulated ARMA(1,1)")
plt.xlim(0,500)
plt.show()

plot_acf(ARMA1,lags=30)
plt.xlim(0,30)
plt.show()

plot_pacf(ARMA1,lags=30,method="ywm")
plt.xlim(0,30)
plt.show()

# =====================================================
# ARMA(2,2)
# =====================================================
arma22_model = ARIMA(X, order=(2,0,2)).fit()

print("\nARMA(2,2) Parameters")
print(arma22_model.params)

phi1 = arma22_model.params["ar.L1"]
phi2 = arma22_model.params["ar.L2"]

theta1 = arma22_model.params["ma.L1"]
theta2 = arma22_model.params["ma.L2"]

ar2 = np.array([1,-phi1,-phi2])
ma2 = np.array([1,theta1,theta2])

arma22 = ArmaProcess(ar2,ma2)
ARMA2 = arma22.generate_sample(nsample=N*10)

plt.figure(figsize=(12,5))
plt.plot(ARMA2)
plt.title("Simulated ARMA(2,2)")
plt.xlim(0,500)
plt.show()

plot_acf(ARMA2,lags=40)
plt.xlim(0,40)
plt.show()

plot_pacf(ARMA2,lags=40,method="ywm")
plt.xlim(0,40)
plt.show()
```
### OUTPUT:
<img width="837" height="391" alt="image" src="https://github.com/user-attachments/assets/898997bf-930b-4631-a7d8-ea1c35cfd0fc" />
<img width="848" height="557" alt="image" src="https://github.com/user-attachments/assets/62500060-c7be-479a-b7e8-56dabdc0d427" />


**SIMULATED ARMA(1,1) PROCESS:**
<img width="1116" height="503" alt="image" src="https://github.com/user-attachments/assets/64e59a23-8b56-4942-963e-33501d87f6b7" />

**Partial Autocorrelation**
<img width="1126" height="603" alt="image" src="https://github.com/user-attachments/assets/d54a904b-3275-4841-80f5-7f353b4080f8" />

**Autocorrelation**
<img width="1130" height="592" alt="image" src="https://github.com/user-attachments/assets/262e2014-13be-4fb3-a010-03012768e2b9" />



**SIMULATED ARMA(2,2) PROCESS:**
<img width="1147" height="682" alt="image" src="https://github.com/user-attachments/assets/223cef35-7550-419c-ae28-112ea6b65078" />

**Partial Autocorrelation**
<img width="1160" height="590" alt="image" src="https://github.com/user-attachments/assets/b7ef962e-6fdf-43d5-abbe-d0520fac1781" />

**Autocorrelation**
<img width="1150" height="602" alt="image" src="https://github.com/user-attachments/assets/347939d2-1621-4e60-9003-cc8c4d09d83b" />

### RESULT:
Thus, a python program is created to fit ARMA Model successfully.
