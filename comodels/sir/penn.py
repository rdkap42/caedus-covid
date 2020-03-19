import numpy as np
from .sir import SIR

#class Penn_basic(SIR):
#    def __init__(self, S: int, I: int, R: int,
#                 icu_rate: float=0.02, hosp_rate: float=0.05,
#                 vent_rate: float=0.01, contact_reduction: float=0.,
#                 t_double: float=6, beta_decay: float=0, vent_los: float=10,
#                 hos_los: float=7, icu_los: float=9, recover_time: float=14) -> None:
#        self.rates = {'hospital': hosp_rate,'icu': icu_rate, 'ventilator':vent_rate}
#        self.los = dict(zip(self.rates.keys(), [hos_los, icu_los, vent_los]))
#        self.contact_reduction = contact_reduction
#        self.t_double = t_double
#        self.intrinsic_growth = 2**(1/t_double) - 1
#        self.recover_time = recover_time
#        gamma = 1/self.recover_time
#        beta = ((self.intrinsic_growth + gamma) / S)  * (1-contact_reduction)
#        self.r_t = beta/gamma * S
#        self.r_naught = self.r_t / (1-contact_reduction)
#        super().__init__(S, I, R, beta, gamma, beta_decay)
#
#    def sir(self, n_days: int) -> dict:
#        s, i, r = super().sir(n_days)
#        out = {}
#        out['infected'] = i
#        out['recovered'] = r
#        out['susceptible'] = s
#        for k in self.rates.keys():
#            out[k] = i*self.rates[k]
#        return out
#
#


def rolling_sum(a: np.ndarray, window: int) -> np.ndarray:
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    rolled = np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)
    return(np.sum(rolled, -1))

class Penn_detect_prob(SIR):
    """
    Penn_detect_prob:
        Make SIR predictions given the assumed detection rate (out of all infected cases), using
        the parameter estimation method from the Penn model.
    ----------------------------------------------------------------------
    Parameters:
        S: pop size, Number of people affected
        I: infected, Number of positive tests in region
        R: recovered, number of recoveries
        detect_prob: percentage of all infected cases in region which you believe are being detected
        hosp_rate: rate of infected admitted to the hospital = 0.05
        icu_rate: rate of infected who need to be in ICU = 0.02
        vent_rate: rate of infected who need ventilators = 0.01
        contact_reduction: percent contact reduced by social distancing = 0
        t_double: time to double number of infected = 6.
        beta_decay: decay rate of beta, which represents how often a contact results in a new infection = 0
        vent_los: time one patient takes up a ventilator=10
        hos_los: time one patient takes up a normal hospital bed = 7
        icu_los: time one patient takes up an ICU bed = 9
        recover_time: time to get better, to shift from I to R = 14

    Attributes:
        rates: dict = rate of hospitalization, icu, and ventilators
        los: dict = length of stay at hospital, icu, and ventilator
        contact_reduction: float = contact reduction
        t_double: float = number of days to double
        intrinsic_growth: float = growth rate
        recover_time: float = time to recover from illness
        beta: float =  how often a contact results in a new infection
        gamma:  float = rate at which an infected person recovers
        beta_decay:  float = decay rate of beta
        r_naught:  float = speadability of disease
        r_t:  float = r_naught after distancing
        S: int = number of susceptible people
        I: int = number of actually infected people
        R: int = number of recovered people


    Methods:
        sir(n_days):
            run simulation
    """
    def __init__(self, S: int, I: int, R: int, detect_prob: float,
                 hosp_rate: float=0.05, icu_rate: float=0.02,
                 vent_rate: float=0.01, contact_reduction: float=0.,
                 t_double: float=6, beta_decay: float=0, vent_los: float=10,
                 hos_los: float=7, icu_los: float=9, recover_time: float=14) -> None:
        self.rates = {'hospital': hosp_rate,'icu': icu_rate, 'ventilator':vent_rate}
        self.los = dict(zip(self.rates.keys(), [hos_los, icu_los, vent_los]))
        self.contact_reduction = contact_reduction
        self.t_double = t_double
        self.intrinsic_growth = 2**(1/t_double) - 1
        self.recover_time = recover_time
        gamma = 1/self.recover_time
        beta = ((self.intrinsic_growth + gamma) / S)  * (1-contact_reduction)
        self.r_t = beta/gamma * S
        self.r_naught = self.r_t / (1-contact_reduction)
        self.I = I / detect_prob
        super().__init__(S, self.I, R, beta, gamma, beta_decay)

    def sir(self, n_days: int) -> (dict, dict):
        s, i, r = super().sir(n_days)
        out = {}
        admissions = {}
        out['infected'] = i
        out['recovered'] = r
        out['susceptible'] = s
        for k in self.rates.keys():
            # calculate raw numbers
            out[k] = i*self.rates[k]
            # turn to new admissions per day
            admissions[k] = out[k]
            admissions[k] = admissions[k][1:] - admissions[k][:-1]
            admissions[k][np.where(admissions[k] < 0)] = 0
            admissions[k] = rolling_sum(admissions[k], self.los[k])
        return out, admissions

