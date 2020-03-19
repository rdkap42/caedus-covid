def get_gamma(t_recover):
    return 1/t_recover

def get_beta_gamma(t_recover, S, t_double = 6, contact_reduction = 0):
    gamma = get_gamma(t_recover)
    intrinsic_growth = 2**(1/t_double) - 1
    beta = ((intrinsic_growth + gamma) / S) * (1-contact_reduction)
    return beta, gamma
