from collections import namedtuple
import numpy as np
import pandas as pd
import random
from scipy.special import gammaln
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
from scipy.linalg import expm

from tqdm import tqdm
from matplotlib import pyplot as plt
from tqdm import tqdm

from eda import us_data
from mass_pop_data import ma_county_pops
from tx_pop_data import tx_county_pops
from nyt_data import county_data
# T = len(us_data['confirmed'])

np.set_printoptions(precision=3)

log = np.log
exp = np.exp

#N = US_POP = 327 * 10**6
underreporting_factors = np.linspace(1, 10, 1000)
doubling_times = np.linspace(2, 7, 1000)

VAR_NAMES = ['s', 'i', 'c', 'ru', 'rc', 'd']
SEIR_VAR_NAMES = ['s', 'e', 'i', 'r', 'd']
SEIR_PARAM_NAMES = ['beta', 'sigma', 'gamma', 'mu', 'I0']
PARAM_NAMES = ['beta', 'delta', 'gamma_u', 'gamma_c', 'mu']
# Param assumptions
incubation_period = 14
recovery_period = 21
fatality_rate = 0.02
R0 = 2.2


iota = 1 / incubation_period
rho = 1 / recovery_period
delta = rho * (fatality_rate) / (1 - fatality_rate)
epsilon = R0 * (rho + delta)

def log_fac(x):
    return gammaln(x + 1)

def sir_deriv(arr, params):
    assert(np.isclose(np.sum(arr), 1))
    s, i, c, ru, rc, d = arr
    beta, delta, gamma_u, gamma_c, mu = params
    ds =  - beta * s * i
    di =  beta * s * i - gamma_u * i - delta * i
    dc = delta * i - (mu + gamma_c) * c
    dru = gamma_u * i
    drc = gamma_c * c
    dd = mu * c
    darr = np.array([ds, di, dc, dru, drc, dd])
    assert(np.isclose(np.sum(darr), 0))
    return darr

def seir_deriv(x, params):
    assert(np.isclose(np.sum(x), 1))
    s, e, i, r, d = x
    beta, sigma, gamma, mu = params
    ds = -beta * s * i
    de = beta * s * i - sigma * e
    di = sigma * e - (gamma + mu) * i
    dr = gamma * i
    dd = mu * i
    dx = np.array([ds, de, di, dr, dd])
    assert(np.isclose(np.sum(dx), 0))
    return dx



def solve_sir(x0, params, end_time):
    f = lambda t, x: sir_deriv(x, params)
    assert(np.isclose(sum(x0), 1))
    t0 = 0
    tf = end_time
    t_span = (t0, tf)
    sol = solve_ivp(f, t_span, x0, max_step=1, t_eval=range(tf))
    return sol

def solve_seir(x0, params, end_time):
    f = lambda t, x: seir_deriv(x, params)
    assert(np.isclose(sum(x0), 1))
    t0 = 0
    tf = end_time
    t_span = (t0, tf)
    sol = solve_ivp(f, t_span, x0, max_step=1, t_eval=range(tf))
    return sol


def init_approximation(params):
    beta, delta, gamma_u, gamma_c, mu = params
    ALPHA = beta - (delta + gamma_u)
    ETA = gamma_c + mu
    coeff = delta * I0/(ALPHA + ETA)
    Kc = -coeff  # c should be zero at t=0
    def c(t):
        return coeff * exp(ALPHA * t) + Kc*exp(-ETA*t)
    def z(t):
        return coeff / ALPHA * exp(ALPHA * t) - Kc / ETA * exp(-ETA*t)
    Kz = -mu * z(0)
    def d(t):
        return mu * z(t) + Kz
    Kru = -gamma_c * z(0)
    def rc(t):
        return gamma_c * z(t) + Kru

    return c, rc, d

def bound(x, N):
    return np.clip(x, 1/N, 1 - 1/N)

def init_approximation_sse(log_params, data):
    M = 10
    N = data['pop']
    T = len(data['confirmed'])
    params = exp(log_params)
    ts = np.arange(T)
    _c, _rc, _d = init_approximation(params)
    c = (lambda x: bound(_c(x)))(ts)[:-2] + 1/N
    d = (lambda x: bound(_d(x)))(ts)[:-2] + 1/N
    rc = (lambda x: bound(_rc(x)))(ts)[:-2] + 1/N
    trash = bound(1 - (c + d + rc))
    obs_c = us_data['confirmed'][:-2]
    obs_d = us_data['deaths'][:-2]
    obs_rc = us_data['recovered']
    obs_trash = N - (obs_c + obs_d + obs_rc)
    prefactor = log_fac(N) - (log_fac(obs_c) + log_fac(obs_d) + log_fac(obs_rc) + log_fac(obs_trash))
    #return sum(((log(c(ts) + 1/N) - log(obs_c + 1/N)))**2) + sum(((log(d(ts) + 1/N) - log(obs_d + 1/N)))**2) + sum((log(rc(ts)[:-2] + 1/N) - log(obs_rc + 1/N))**2)
    return sum(prefactor + obs_c * log(c) + obs_d * log(d) + obs_rc * log(rc) + obs_trash * log(trash))

def q(x, sigma=0.01):
    """for use with log params"""
    return x + np.random.normal(0, sigma, size=len(x))

def mh(lf, q, x, iterations=10000, modulus=100):
    traj = []
    ll = lf(x)
    accepts = 0
    for iteration in range(iterations):
        xp = q(x)
        llp = lf(xp)
        if log(random.random()) < llp - ll:
            x = xp
            ll = llp
            accepts += 1
        if iteration % modulus == 0:
            traj.append((x, ll))
            print(
                "{}/{} log_params: {} log-likelihood: {:1.3f} acceptances: {} acceptance ratio: {:1.3f}".format(
                    iteration, iterations, x, ll, accepts, accepts / (iteration + 1)
                )
            )
    return traj

def fit_init_approximation(tol=10**-14):
    x0 = np.random.normal(0, 1, size=len(PARAM_NAMES))
    # x0 = np.array([ 13.26726095,  -7.21161112,  13.26726049,  -6.55617211,
    #    -52.65910809])
    return minimize(init_approximation_sse, x0, method='powell', options={'maxiter': 100000, 'xtol':tol, 'disp':True})

def check_init_approxiation_fit(tol):
    sol = fit_init_approximation(tol)

def plot_log_params(log_params, data, plot_data=True, plot_legend=True, show=True):
    params = exp(log_params)
    N = data['pop']
    T = len(data['confirmed'])
    c, rc, d = init_approximation(params)
    obs_c = data['confirmed'] / N
    obs_d = data['deaths'] / N
    obs_rc = data['recovered'] / N
    ts = np.arange(T)
    if plot_data:
        plt.plot(obs_c, linestyle=' ', marker='o', label='obs c')
        plt.plot(obs_d, linestyle=' ', marker='o', label='obs d')
        plt.plot(obs_rc, linestyle=' ', marker='o', label='obs rc')
    plt.plot(c(ts), label='est c', color='b', linestyle='--')
    plt.plot(d(ts), label='est d', color='orange', linestyle='--')
    plt.plot(rc(ts), label='est rc', color='g', linestyle='--')
    if plot_legend:
        plt.legend()
    if show:
        plt.show()


def test_init_approximation(data):
    # VAR_NAMES = ['s', 'i', 'c', 'ru', 'rc', 'd']
    N = data['pop']
    I0 = 1/N
    ic = [1-I0, I0, 0, 0, 0, 0]
    params = np.array([ 0.82,  0.22,  0.34,  2.30, 10.28]) * 3
    sol = solve_sir(ic, params)

def estimate_init_conds():
    confirmed_cases = 13
    underreporting_factor = 10
    initial_cases = confirmed_cases * underreporting_factor
    susceptible_cases = boston_pop - initial_cases
    infected_cases = initial_cases / 3
    exposed_cases = initial_cases - infected_cases
    s = susceptible_cases / boston_pop
    e = exposed_cases / boston_pop
    i = infected_cases / boston_pop
    d = 0
    r = 0

def plot_sir_sol(sol):
    ts = sol.t
    c = sol.y[VAR_NAMES.index('c'), :]
    i = sol.y[VAR_NAMES.index('i'), :]
    y = c + i
    y0, yf = y[0], y[10]
    t0, tf = ts[0], ts[10]
    doublings = np.log2(yf / y0)
    doubling_time = (tf - t0) / doublings
    print("doubling time:", doubling_time)
    for i, var_name in enumerate(var_names):
        plt.plot(sol.y[i, :], label=var_name)
    plt.legend()
    plt.show()

def log_likelihood(sol, data):
    obs_c = data['confirmed']
    obs_rc = data['recovered']
    obs_d = data['deaths']
    N = data['pop']
    T = len(data['confirmed'])
    y_c = sol.y[VAR_NAMES.index('c'), :]
    #y_rc = sol.y[VAR_NAMES.index('rc'), :]
    y_d = sol.y[VAR_NAMES.index('d'), :]
    y_trash = 1 - (y_c + y_d)
    log_prob = 0
    for t in range(T):
        #print(t)
        C, D = obs_c[t], obs_d[t]
        TRASH = N - (C + D)
        c, d, trash = y_c[t], y_d[t], y_trash[t]
        prefactor = log_fac(N) - (log_fac(C) + log_fac(D) + log_fac(TRASH))
        #print(c, rc, d)
        log_prob_t = prefactor + C * log(c) + D * log(d) + TRASH * log(trash)
        #print(prefactor, log_prob_t)
        log_prob += log_prob_t
    return log_prob

def log_likelihood2(sol, data):
    obs_c = data['confirmed']
    obs_rc = data['recovered']
    obs_d = data['deaths']
    N = data['pop']
    T = len(data['confirmed'])
    y_c = sol.y[VAR_NAMES.index('c'), :]
    y_rc = sol.y[VAR_NAMES.index('rc'), :]
    y_d = sol.y[VAR_NAMES.index('d'), :]
    y_trash = 1 - (y_c + y_rc + y_d)
    log_prob = 0
    for t in range(T):
        #print(t)
        C, RC, D = obs_c[t], obs_rc[t], obs_d[t]
        TRASH = N - (C + RC + D)
        c, rc, d, trash = y_c[t], y_rc[t], y_d[t], y_trash[t]
        #print(c, rc, d)
        log_prob_t = -((C - c*N)**2 + (RC - rc*N)**2 + (D - (d*N))**2 + (TRASH - trash*N)**2)
        #print(prefactor, log_prob_t)
        log_prob += log_prob_t
    return log_prob

def seir_log_likelihood(sol, data, only_deaths=True):
    obs_c = data['confirmed']
    obs_d = data['deaths']
    N = data['pop']
    T = len(data['confirmed'])
    y_c = bound(sol.y[SEIR_VAR_NAMES.index('i'), :], N)
    y_d = bound(sol.y[SEIR_VAR_NAMES.index('d'), :], N)
    if only_deaths:
        y_trash = 1 - (y_d)
    else:
        y_trash = 1 - (y_c + y_d)
    log_prob = 0
    for t in range(T):
        #print(t)
        # if obs_c[t] < 100:
        #     continue
        if only_deaths:
            D = obs_d[t]
            TRASH = N - D
            d, trash = y_d[t], y_trash[t]
            log_prob += multinomial_ll([d, trash], [D, TRASH])
        else:
            C, D = obs_c[t], obs_d[t]
            TRASH = N - (C + D)
            c, d, trash = y_c[t],  y_d[t], y_trash[t]
            log_prob += multinomial_ll([c, d, trash], [C, D, TRASH])
        # log_prob += sse_ll([c, d, trash], [C, D, TRASH])
    return log_prob

def multinomial_ll(ps, obs):
    N = np.sum(obs)
    prefactor = log_fac(N) - sum(log_fac(n) for n in obs)
    return prefactor + sum(o * log(p) for (p, o) in zip(ps, obs))

def sse_ll(ps, obs):
    N = sum(obs)
    return -sum((p * N - o)**2 for (p, o) in zip(ps, obs))


def random_hyp():
    ic = np.array([0.99] + [random.random() * 0.01 for _ in range(len(VAR_NAMES) - 1)])
    ic = ic / sum(ic)

    log_thetas = np.random.normal(0, 1, size=len(PARAM_NAMES))
    thetas = exp(log_thetas)
    thetas[5:] /= 10
    return ic, thetas

def mutate_hyp(hyp):
    ic, thetas = hyp
    log_ic = log(ic)
    new_log_ic = log_ic + np.random.normal(0, 0.01, size=len(ic))
    new_ic = exp(new_log_ic)
    new_ic /= sum(new_ic)
    log_thetas = log(thetas)
    new_log_thetas = log_thetas + np.random.normal(0, 0.01, size=len(thetas))
    new_thetas = exp(new_log_thetas)
    return new_ic, new_thetas

def ll_from_hyp(hyp, data):
    ic, thetas = hyp
    T = len(data['confirmed'])
    sol = solve_sir(ic, thetas, T)
    return log_likelihood(sol, data)

def fit_model(data, generations=10000):
    ll = None
    traj = []
    acceptances = 0
    while ll is None:
        hyp = random_hyp()
        print(hyp)
        prop_ll = ll_from_hyp(hyp, data)
        if not np.isnan(prop_ll):
            ll = prop_ll
    for t in range(generations):
        hyp_p = mutate_hyp(hyp)
        ll_p = ll_from_hyp(hyp_p, data)
        if np.log(random.random()) < ll_p - ll:
            acceptances += 1
            hyp = hyp_p
            ll = ll_p
        if t % 100 == 0:
            traj.append((hyp, ll))
            print(t, ll, "ar:", acceptances / (t + 1))
            print(hyp)
    return traj

def ps_from_lls(lls):
    print("min, max:", min(lls), max(lls))
    a = min(lls)
    expa = exp(a)
    ps = [exp(ll - a) for ll in lls]
    return ps

def check_hyp(hyp, data):
    N = data['pop']
    T = len(data['confirmed'])
    x0, params = hyp
    sol = solve_sir(x0, params, T)
    for name, ts in zip(VAR_NAMES, sol.y):
        plt.plot(ts, label=name)
    plt.plot(data['confirmed'] / N, label='obs confirmed', marker='o', linestyle=' ')
    plt.plot(data['recovered'] / N, label='obs recovered', marker='o', linestyle=' ')
    plt.plot(data['deaths'] / N, label='obs deaths', marker='o', linestyle=' ')
    plt.legend()


def plot_lls(traj):
    lls = [ll for (x, ll) in traj]
    plt.subplot(2, 1, 1)
    plt.plot(lls)
    plt.xlabel("Iterations x 100", size='x-large')
    plt.ylabel("Log-likelihood", size='x-large')
    plt.subplot(2, 1, 2)
    plt.plot(lls)
    plt.ylim(-760, -730)
    plt.xlabel("Iterations x 100", size='x-large')
    plt.ylabel("Log-likelihood", size='x-large')
    plt.tight_layout()
    plt.savefig("ll-plot.png", dpi=300)

def plot_param_results(traj, data):
    """Use with SIR"""
    N = data['pop']
    T = len(data['confirmed'])
    log_params, ll = traj[-1]
    params = exp(log_params)
    # VAR_NAMES = ['s', 'i', 'c', 'ru', 'rc', 'd']
    params = exp(log_params)
    c, rc, d = init_approximation(params)

    sir_x0 = np.array([1-1/N, 1/N, 0, 0, 0, 0])
    sir_sol = solve_sir(sir_x0, params)
    sir_c, sir_rc, sir_d = sir_sol.y[2], sir_sol.y[4], sir_sol.y[5]

    obs_c = data['confirmed'] / N
    obs_d = data['deaths'] / N
    obs_rc = data['recovered'] / N
    ts = np.arange(T)


    plt.subplot(3, 1, 1)
    plt.plot(obs_c, linestyle=' ', marker='o', label='C (observed)')
    plt.plot(sir_c,  color='blue', label='C (SIR model)')
    plt.plot(c(ts),  color='orange', linestyle='--', label='C (init approx)')
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(obs_rc, linestyle=' ', marker='o', label='Rc (observed)')
    plt.plot(sir_rc,  color='blue', label='Rc (SIR model)')
    plt.plot(rc(ts),  color='orange', linestyle='--', label='Rc (init approx)')
    plt.ylabel("Population Fraction", size='x-large')
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.plot(obs_d, linestyle=' ', marker='o', label='D (observed)')
    plt.plot(sir_d,  color='blue', label='D (SIR model)')
    plt.plot(d(ts),  color='orange', linestyle='--', label='D (init approx)')
    plt.legend()
    plt.xlabel("Days since 1/22/20", size='x-large')

    plt.tight_layout()
    plt.savefig("fit-results.png", dpi=300)
    plt.close()

def log_param_scatterplot(log_param_traj, param_names=["beta", "sigma", "gamma", "mu", "IO"]):
    # param_traj = [exp(lp) for lp in log_param_traj]
    K = len(log_param_traj[0])
    log_param_vecs = list(zip(*log_param_traj))
    for i, i_param in enumerate(param_names):
        for j, j_param in enumerate(param_names):
            plt_idx = j * K + i + 1
            print(i_param, j_param)
            plt.subplot(K, K, plt_idx)
            if plt_idx % K == 1:
                plt.ylabel(j_param)
            if j_param == param_names[-1]:
                plt.xlabel(i_param)
                print("x label:", i_param)

            #plt.title(i_param + " " + j_param)
            if i == j:
                plt.hist(log_param_vecs[i])
            else:
                plt.scatter(log_param_vecs[i], log_param_vecs[j], s=5)
    # plt.tight_layout()
    # plt.savefig("param-pairplots.png", dpi=300)
    # plt.close()l
def seir_experiment(data, log_params=None, iterations=10_000, sigma=0.01, only_deaths=True):
    # S, E, I, R, D

    T = len(data['confirmed'])
    if log_params is None:
        log_params = np.array([-0.19780107, -2.65762238, -3.21675428, -6.12722099, -19.6])
        log_params = np.random.normal(-2, 1, size=len(log_params))
        N = data['pop']
        I0 = 1/N
        log_params[-1] = log(I0)  # seed I0 as 1 / US_POP

    def lf(log_params):
        params = exp(log_params)
        params, I0 = params[:-1], params[-1]
        init_condition = np.array([1 -I0, 0, I0, 0, 0])
        sol = solve_seir(init_condition, params, T)
        return seir_log_likelihood(sol, data, only_deaths=only_deaths)
    traj = mh(lf, lambda x:q(x, sigma=sigma), log_params, modulus=10, iterations=iterations)
    return traj
    # log_params1 = traj1[-1][0]
    # traj01 = mh(lf, lambda x:q(x, sigma=0.01), log_params1, modulus=10, iterations=1000)
    # log_params01 = traj01[-1][0]
    # traj001 = mh(lf, lambda x:q(x, sigma=0.01), log_params01, modulus=10, iterations=1000)
    # log_params001 = traj001[-1][0]
    # traj0001 = mh(lf, lambda x:q(x, sigma=0.001), log_params001, modulus=10, iterations=1000)
    # return traj1 + traj01 + traj--1 + traj0001


def plot_seir_param_results(traj, data, fname=None):
    log_params, ll = traj[-1]
    T = len(data['confirmed'])
    params = exp(log_params)
    # SEIRD
    params, I0 = params[:-1], params[-1]
    init_condition = np.array([1 -I0, 0, I0, 0, 0])
    seir_sol = solve_seir(init_condition, params, T)
    seir_c, seir_r, seir_d = seir_sol.y[2], seir_sol.y[3], seir_sol.y[4]

    N = data['pop']
    T = len(data['confirmed'])
    obs_c = data['confirmed']
    obs_d = data['deaths']
    #obs_rc = data['recovered'] / N
    ts = np.arange(T)

    approx_f = seir_approximation(init_condition, params)
    approx_c = np.array([approx_f(t)[SEIR_VAR_NAMES.index('i')] for t in ts])
    #approx_r = [approx_f(t)[SEIR_VAR_NAMES.index('r')] for t in ts]
    approx_d = np.array([approx_f(t)[SEIR_VAR_NAMES.index('d')] for t in ts])

    plt.subplot(2, 1, 1)
    plt.plot(obs_c, linestyle=' ', marker='o', label='C (observed)')
    plt.plot(seir_c * N,  color='blue', label='C (SEIR model)')
    plt.plot(approx_c * N,  color='orange', label='C (approx)', linestyle='--')
    # for log_params, ll in traj[::10]:
    #     params = exp(log_params)
    #     params, I0 = params[:-1], params[-1]
    #     init_condition = np.array([1 -I0, 0, I0, 0, 0])
    #     seir_sol = solve_seir(init_condition, params)
    #     seir_c, seir_r, seir_d = seir_sol.y[2], seir_sol.y[3], seir_sol.y[4]
    #     plt.plot(seir_c,  color='blue', alpha=0.01)
    plt.legend()

    # plt.subplot(3, 1, 2)
    # plt.plot(obs_rc, linestyle=' ', marker='o', label='Rc (observed)')
    # plt.plot(seir_r,  color='blue', label='Rc (SEIR model)')
    # plt.plot(approx_r,  color='orange', label='Rc (approx)', linestyle='--')
    # plt.ylabel("Population Fraction", size='x-large')
    # plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(obs_d, linestyle=' ', marker='o', label='D (observed)')
    plt.plot(seir_d * N,  color='blue', label='D (SEIR model)')
    plt.plot(approx_d * N,  color='orange', label='D (approx)', linestyle='--')
    plt.legend()
    plt.xlabel("Days since 1/22/20", size='x-large')

    if fname:
        plt.tight_layout()
        plt.savefig("fit-results.png", dpi=300)
        plt.close()
    else:
        plt.show()

def plot_seir_sol(sol):
    start = 'Jan 22, 2020'
    today = pd.to_datetime('now')
    date_range = pd.date_range(start=start, end=today)
    today_t = len(date_range)
    for var_name, data in zip(SEIR_VAR_NAMES, sol.y):
        plt.plot(data, label=var_name)
    plt.axvline(today_t, linestyle='--', label='today')
    plt.legend()
    plt.xlabel("Days since 1/22/2020")
    plt.show()

def plot_seir_sols_from_traj(traj, data):
    N = data['pop']
    colors = 'brygc'
    for i, (log_params, ll) in tqdm(enumerate(traj)):
        params = exp(log_params)
        params, I0 = params[:-1], params[-1]
        init_condition = np.array([1 -I0, 0, I0, 0, 0])
        sol = solve_seir(init_condition, params, 365*2)
        for var_name, time_series, color in zip(SEIR_VAR_NAMES, sol.y, colors):
            plt.plot(
                time_series * N,
                label=(var_name if i == 0 else None),
                color=color,
                alpha=0.5
            )
    plt.plot(data['confirmed'], marker='o', linestyle=' ', label='obs C')
    plt.plot(data['deaths'], marker='o', linestyle=' ', label='obs D')
    start = 'Jan 22, 2020'
    today = pd.to_datetime('now')
    date_range = pd.date_range(start=start, end=today)
    today_t = len(date_range)
    plt.axvline(today_t, linestyle='--', label='today')
    plt.legend()
    plt.xlabel("Days since 1/22/2020")
    plt.show()

def seir_approximation(y0, params):
    beta, sigma, gamma, mu = params
    Gamma = gamma + mu
    A = np.matrix([
        [0, 0, -beta, 0, 0],
        [0, -sigma, beta, 0, 0],
        [0, sigma, -(gamma + mu), 0, 0],
        [0, 0, gamma, 0, 0],
        [0, 0, mu, 0, 0],
    ])
    return lambda t: expm(A*t).dot(y0)

def make_csv_from_ma_traj(traj, data, fname):
    N = data['pop']
    T = 365
    cases = [[] for _ in range(T)]
    deaths = [[] for _ in range(T)]
    for log_tunables, ll in tqdm(traj):
        log_params, log_I0 = log_tunables[:-1], log_tunables[-1]
        params, I0 = exp(log_params), exp(log_I0)
        x0 = np.array([1 -I0, 0, I0, 0, 0])
        sol = solve_seir(x0, params, end_time=T)
        cases_idx = SEIR_VAR_NAMES.index('i')
        deaths_idx = SEIR_VAR_NAMES.index('d')
        num_cases = sol.y[cases_idx, :] * N
        num_deaths = sol.y[deaths_idx, :] * N
        for i in range(T):
            cases[i].append(num_cases[i])
            deaths[i].append(num_deaths[i])
        # cases = [sorted(col) for col in cases]
        # deaths = [sorted(col) for col in deaths]
    cases_mean = [np.mean(col) for col in cases]
    cases_2p5 = [np.percentile(col, 2.5) for col in cases]
    cases_97p5 = [np.percentile(col, 97.5) for col in cases]
    deaths_mean = [np.mean(col) for col in deaths]
    deaths_2p5 = [np.percentile(col, 2.5) for col in deaths]
    deaths_97p5 = [np.percentile(col, 97.5) for col in deaths]

    start = 'Jan 22, 2020'
    end = pd.to_datetime(start) + pd.Timedelta(days=(365 - 1))
    date_range = pd.date_range(start=start, end=end)
    data_dict = {
        'Date': date_range,
        'Cases_Mean': round_to_int(cases_mean),
        'Cases_LB': (round_to_int(cases_2p5)),
        'Cases_UB': (round_to_int(cases_97p5)),
        'Deaths_Mean': (round_to_int(deaths_mean)),
        'Deaths_LB': (round_to_int(deaths_2p5)),
        'Deaths_UB': (round_to_int(deaths_97p5)),
        }
    data_cols = ['Cases_Mean', 'Cases_LB', 'Cases_UB', 'Deaths_Mean', 'Deaths_LB', 'Deaths_UB']
    for county, county_pop in sorted(ma_county_pops.items()):
        county_frac = county_pop / N
        for col_name in data_cols:
            col = data_dict[col_name]
            county_col = round_to_int(col * county_frac)
            county_col_name = county + "_" + col_name
            data_dict[county_col_name] = county_col
    df = pd.DataFrame(data_dict)
    df.set_index('Date')
    df.to_csv(fname, index=False)

def make_csv_from_tx_traj(traj, data, fname):
    N = data['pop']
    T = 365
    cases = [[] for _ in range(T)]
    deaths = [[] for _ in range(T)]
    for log_tunables, ll in tqdm(traj):
        log_params, log_I0 = log_tunables[:-1], log_tunables[-1]
        params, I0 = exp(log_params), exp(log_I0)
        x0 = np.array([1 -I0, 0, I0, 0, 0])
        sol = solve_seir(x0, params, end_time=T)
        cases_idx = SEIR_VAR_NAMES.index('i')
        deaths_idx = SEIR_VAR_NAMES.index('d')
        num_cases = sol.y[cases_idx, :] * N
        num_deaths = sol.y[deaths_idx, :] * N
        for i in range(T):
            cases[i].append(num_cases[i])
            deaths[i].append(num_deaths[i])
        # cases = [sorted(col) for col in cases]
        # deaths = [sorted(col) for col in deaths]
    cases_mean = [np.mean(col) for col in cases]
    cases_2p5 = [np.percentile(col, 2.5) for col in cases]
    cases_97p5 = [np.percentile(col, 97.5) for col in cases]
    deaths_mean = [np.mean(col) for col in deaths]
    deaths_2p5 = [np.percentile(col, 2.5) for col in deaths]
    deaths_97p5 = [np.percentile(col, 97.5) for col in deaths]

    start = 'Jan 22, 2020'
    end = pd.to_datetime(start) + pd.Timedelta(days=(365 - 1))
    date_range = pd.date_range(start=start, end=end)
    data_dict = {
        'Date': date_range,
        'Cases_Mean': round_to_int(cases_mean),
        'Cases_LB': (round_to_int(cases_2p5)),
        'Cases_UB': (round_to_int(cases_97p5)),
        'Deaths_Mean': (round_to_int(deaths_mean)),
        'Deaths_LB': (round_to_int(deaths_2p5)),
        'Deaths_UB': (round_to_int(deaths_97p5)),
        }
    data_cols = ['Cases_Mean', 'Cases_LB', 'Cases_UB', 'Deaths_Mean', 'Deaths_LB', 'Deaths_UB']
    for county, county_pop in sorted(tx_county_pops.items()):
        county_frac = county_pop / N
        for col_name in data_cols:
            col = data_dict[col_name]
            county_col = round_to_int(col * county_frac)
            county_col_name = county + "_" + col_name
            data_dict[county_col_name] = county_col
    df = pd.DataFrame(data_dict)
    df.set_index('Date')
    df.to_csv(fname, index=False)

def check_csv_from_traj():
    df = pd.read_csv("seir_output.csv")
    print(df)


def round_to_int(x):
    return np.array(np.round(x), dtype=int)









def make_csv_from_sol(log_params):
    BOSTON_POP = 4.6 * 10**6
    SEVERE_FRACTION = 0.1
    ICU_FRACTION = 0.05
    params = exp(log_params)
    params, I0 = params[:-1], params[-1]
    init_condition = np.array([1 -I0, 0, I0, 0, 0])

    start = 'Jan 22, 2020'
    today = pd.to_datetime('now')
    end = today + pd.Timedelta(days=30)
    date_range = pd.date_range(start=start, end=end)

    seir_sol = solve_seir(init_condition, params, end_time=len(date_range))
    cases_idx = SEIR_VAR_NAMES.index('i')
    deaths_idx = SEIR_VAR_NAMES.index('d')
    cases_fraction = seir_sol.y[cases_idx, :]
    deaths_fraction = seir_sol.y[deaths_idx, :]
    boston_cases = (BOSTON_POP * cases_fraction)
    boston_deaths = BOSTON_POP * deaths_fraction
    cases = [int(round(c)) for c in boston_cases]
    severe_cases = [int(round(c)) for c in SEVERE_FRACTION * boston_cases]
    icu_cases = [int(round(c)) for c in ICU_FRACTION * boston_cases]
    deaths = [int(round(d)) for d in boston_deaths]

    df = pd.DataFrame({
        'Date': date_range,
        'TotalCases': cases,
        'SevereCases': severe_cases,
        'ICUCases': icu_cases,
        'Deaths': deaths
    })
    df.set_index('Date')
    df.to_csv("mock_output.csv", index=False)
    pass

def inspect_traj(traj):
    plt.subplot(1, 2, 1)
    plt.plot([ll for (x, ll) in traj])
    plt.subplot(1, 2, 2)
    plt.plot([x for (x, ll) in traj])
    plt.show()
    log_param_scatterplot([x for (x, ll) in traj])
    plt.show()
    thetas = [theta for (theta, ll) in traj]
    params = zip(*thetas)
    for param_name, param in zip(SEIR_PARAM_NAMES, params):
        coef_of_var = np.std(param)/abs(np.mean(param))
        print(param_name, coef_of_var)

def fit_state(state_name):
    counties_data = {count}

def sigmoid(x, a, b):
    return 1 / (1 + exp(a*x + b))
def sigmoid_fit(ys, iterations=10000):
    def f(a_b):
        a, b = a_b
        return -log(sum((y - sigmoid(t, a, b))**2 for (t, y) in enumerate(ys)))
    def q(a_b):
        return a_b + np.random.normal(size=2)
    x0 = np.array([100, 100])
    traj = mh(f, q, x0, iterations=iterations)
    return traj
