from comodels.sir import Penn_detect_prob
from comodels.param_estimation import get_beta_gamma
import matplotlib as mpl
mpl.use('tkagg') # qt is broken on my computer
import matplotlib.pyplot as plt

# model of TX
S = 28304596

# https://dshs.texas.gov/news/updates.shtm#coronavirus
# csse says otherwise, i will use the bigger number
I = 223

# doubling rate:
# https://www.reddit.com/r/COVID19/comments/fk1zbo/current_us_covid19_confirmed_case_doubling_rate_3/
# https://imgur.com/gallery/7EZKVph
t_double = 3

# recoveries or deaths (cannot be infected by disease anymore)
R = 3

# rate of hospitalization, ICU,  ventilators(guess)
# https://www.statnews.com/2020/03/10/simple-math-alarming-answers-covid-19/
hosp_rate = 0.15
icu_rate = 0.05
vent_rate = 0.02

# length of stay for ventilator and hospital and icu, taken from penn
hos_los = 7
icu_los = 9
vent_los = 10


# time to get better or die, according to italy report it is about 9 days from
# the onset of symptoms. We will assume it is the length of quarantine, or 14
# days. We will estimate this in the future
t_recover = 14

# social distancing parameter. This will be estimated in the future. TX just
# closed all bars restaurants etc:
contact_reduction = 0.1

# we will assume detection probability is 1, that is if you are sick it is
# known. For vanilla SIR models, we will see that this actually doesnt matter
# for practical purposes
detect_prob = 1
print("beta: {}, gamma: {}".format(*get_beta_gamma(t_recover, S, contact_reduction)))

# reduction in spreadability over time, we will assume is 0
beta_decay = 0

print(Penn_detect_prob.__doc__)

tx = Penn_detect_prob(S, I, R, detect_prob, hosp_rate, icu_rate, vent_rate,
                      contact_reduction, t_double, beta_decay,
                      vent_los, hos_los, icu_los, t_recover)

# get the curve and (I believe) delta occupancy
curve, occupancy = tx.sir(140)

good_test_dict = {k: max(v) for k, v in occupancy.items()}

def plot_penn(Pdp: Penn_detect_prob, n_days: int) -> None:
    curve, admissions = Pdp.sir(n_days)
    fig, ax = plt.subplots(1,3, figsize=(15,5))
    for k, v in curve.items():
        if k not in tx.rates.keys():
            ax[0].plot(v, label=k)
            ax[0].legend()
        else:
            ax[1].plot(v, label=k)
            ax[1].legend()
    ax[1].set_title('Hospital Resource Usage')
    ax[0].set_title('SIR curve')
    for k, v in admissions.items():
        ax[2].plot(v, label = k)
        ax[2].legend()
    ax[2].set_title('Change in hospital occupancy')
    plt.show()

plot_penn(tx, 140)

# the curve just shifts to the left
tx_bad_testing = Penn_detect_prob(S, I, R, 0.5, hosp_rate, icu_rate, vent_rate,
                      contact_reduction, t_double, beta_decay,
                      vent_los, hos_los, icu_los, t_recover)
plot_penn(tx_bad_testing, 140)

bad_test_dict = {k: max(v) for k, v in tx_bad_testing.sir(140)[-1].items()}

print([good_test_dict[k] - bad_test_dict[k] for k in bad_test_dict.keys()])
