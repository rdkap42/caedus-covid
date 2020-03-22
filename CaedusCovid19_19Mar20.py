import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt


# This code was pulled from https://scipython.com/book/chapter-8-scipy/additional-examples/the-sir-epidemic-model/
#**********************************************************************************
# Area of regard, A (sq mi).
A = 89.63 # Boston Area
# Population Density, rho (people/sq mi)
rho = 13841 # (2020 data)

# Total population, N.
#N = 1000
N = rho
# Initial number of infected and recovered individuals, I0 and R0.
I0, R0 = 1, 0
# Everyone else, S0, is susceptible to infection initially.
S0 = N - I0 - R0
# Contact rate, beta, and mean recovery rate, gamma, (in 1/days).
beta, gamma = 0.2, 1./10
# A grid of time points (in days)
t = np.linspace(0, 180, 181)

# The SIR model differential equations.
def deriv(y, t, N, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I
    return dSdt, dIdt, dRdt


# Initial conditions vector
y0 = S0, I0, R0
# Integrate the SIR equations over the time grid, t.
ret = odeint(deriv, y0, t, args=(N, beta, gamma))
S, I, R = ret.T


def cumulative_hosp_stats(I, rate_dict = {'Hospital Beds': 0.15, 'ICU Beds': 0.05, 'Ventilators': 0.02}):
    return {k: I*v for k, v in rate_dict.items()}


def calc_admissions_change(admits):
    # change in admissions (decumulating)
    out = admits[1:] - admits[:-1]
    out[np.where(out < 0)] = 0
    return out.copy()


def rolling_sum(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    rolled = np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)
    return rolled.sum(-1).copy()


def plot_dict(d, title):
    fig = plt.figure()
    ax = fig.add_subplot()
    for k in d.keys():
        ax.plot(d[k], label = k)
        ax.legend()
    ax.set_title(title)
    ax.set_xlabel('Days from today')



# CUMULATIVE HOSPITAL USAGE PLOT
usage = cumulative_hosp_stats(I)
plot_dict(usage, "Cumulative Hospital Usage")
plt.show()


# DAILY ADMISSIONS PLOT
admissions = {k: calc_admissions_change(v) for k, v in usage.items()}
plot_dict(admissions, "Daily Hospital Admissions")
plt.show()


# HOSPITAL OCCUPANCY CENSUS
los_dict = dict(zip(admissions.keys(), [7,9,10]))
census = {k: rolling_sum(v, los_dict[k]) for k, v in admissions.items()}
plot_dict(census, "Census of Hospital Occupancy")
plt.show()




# Plot the data on three separate curves for S(t), I(t) and R(t)
#fig = plt.figure(facecolor='w')
#ax = fig.add_subplot(111, axis_bgcolor='#dddddd', axisbelow=True)
#ax.plot(t, S, 'b', alpha=0.5, lw=2, label='Susceptible')
#ax.plot(t, I, 'r', alpha=0.5, lw=2, label='Infected')
#ax.plot(t, R, 'g', alpha=0.5, lw=2, label='Recovered')
#ax.set_xlabel('Time /days')
#ax.set_ylabel('Number /sq mi')
##ax.set_ylim(0,1.2)
#ax.yaxis.set_tick_params(length=0)
#ax.xaxis.set_tick_params(length=0)
#ax.grid(b=True, which='major', c='w', lw=2, ls='-')
#legend = ax.legend()
#plt.text(23, 5300, r'contact rate: '+ str(beta), {'color': 'black', 'fontsize': 16})
#plt.text(23, 4400, r'recovery rate: '+ str(gamma), {'color': 'black', 'fontsize': 16})
#plt.text(int(t[I==max(I)])-1, max(I)+100, r''+ maxI, {'color': 'red', 'fontsize': 16})
#legend.get_frame().set_alpha(0.5)
#for spine in ('top', 'right', 'bottom', 'left'):
#    ax.spines[spine].set_visible(False)
#plt.show()
##**********************************************************************************
