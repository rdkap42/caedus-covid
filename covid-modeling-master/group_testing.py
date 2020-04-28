import numpy as np

from tqdm import *

log = np.log
log10 = np.log10

def group_test(n, p):
    return (np.random.random(size=n) < p).any()

def log_likelihood(result, n, p):
    if result:
        return log(1 - (1 - p)**n)
    else:
        return n * log(1 - p)

def group_log_likelihood(data, p):
    ll = 0
    for result, n in data:
        ll += log_likelihood(result, n, p)
    return ll

def calc_optimal_batch_size(p):
    """1 - (1-p)**n == 1/2"""
    # 1/2 = (1-p)**n
    # -log(2) = n log(1 - p)
    return int(round(log(2) / log(1/(1 - p))))

def simulate(true_p, ns):
    # trials = 10
    data = [(group_test(n, true_p), n) for n in ns]
    return data
    # ps = np.logspace(-6, -0.01)
    # lls = [group_log_likelihood(data, p) for p in ps]
    # plt.plot(ps, lls)
    # plt.semilogx()

def mh(data, trials=1000):
    log_theta = log(0.5)
    traj = [exp(log_theta)]
    def log_f(log_theta):
        return group_log_likelihood(data, exp(log_theta))
    ll = log_f(log_theta)
    def q(log_theta):
        return min(log_theta + np.random.normal(), 0 - 10**-6)
    for trial in range(trials):
        log_thetap = q(log_theta)
        llp = log_f(log_thetap)
        if log(random.random()) < llp - ll:
            log_theta = log_thetap
            ll = llp
        traj.append(exp(log_theta))
        if trial % 100 == 0: pass
            #print(trial, exp(log_theta), ll)
    return traj

def run_sim(true_p=0.001):
    #ns = [1, 2, 4, 8, 16, 32, 64]
    ns = [14] * 7
    data = simulate(true_p, ns)
    traj = mh(data, trials=10000)
    ps = np.logspace(-6, 0 - 10**-6, 1000)
    lls = [group_log_likelihood(data, p) for p in ps]
    # plt.subplot(1, 2, 1)
    # plt.plot(ps, lls)
    # plt.ylim(-20, -5)
    # plt.axvline(true_p, linestyle='--')
    # plt.xlabel("Base Rate of Infection")
    # plt.ylabel("Log-Likelihood")
    # plt.semilogx()
    # plt.subplot(1, 2, 2)
    # plt.hist(traj[100:], bins=100)
    #plt.show
    pred = exp(np.mean(log(traj)))
    return (pred < 0.05) == (true_p < 0.05)
    # print("log pred:", np.mean(log10(traj)), np.std(log10(traj)))
    # print("pred:", np.mean((traj)), np.std((traj)))
    # print(np.mean([abs((true_p - pred_p)/true_p) for pred_p in traj]))
    # print(ci(traj))
    # print(ci(log10(traj)))


def do_sims():
    ps = exp(np.linspace(log(1/1000), 0, 100))
    sim_results = []
    for trial in trange(100):
        true_p = random.choice(ps)
        sim_results.append(run_sim(true_p))
        print(np.mean(sim_results))
    return sim_results
def ci(xs):
    mu = np.mean(xs)
    sigma = np.std(xs)
    return (mu - 1.96 * sigma, mu + 1.96 * sigma)
