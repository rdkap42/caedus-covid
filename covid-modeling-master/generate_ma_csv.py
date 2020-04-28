import logging
import time

import numpy as np

from eda import ma_data, tx_data
from sir_fitting_us import seir_experiment, make_csv_from_tx_traj

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

logger.info("Fitting model.")

# initial values taken from previous fit, used to seed MH sampler efficiently.
x0 = np.array([  0.393,  -2.586,  -3.241,  -5.874, -24.999])
# ma_traj = seir_experiment(ma_data, x0, iterations=10000)
tx_traj = seir_experiment(tx_data, x0, iterations=10000)

# mean_ll = np.mean([ll for (x, ll) in ma_traj])
mean_ll = np.mean([ll for (x, ll) in tx_traj])
logger.info("Model fitting finished with mean log-likelihood: {}".format(mean_ll))

if mean_ll < -2000:
    raise AssertionError(
        """Mean log-likelihood {} less than threshold of
        -20.  This is probably an error.""".format(mean_ll)
    )

underscored_time = time.ctime().replace(" ", "_")
fname = "ma_seir_output_{}.csv".format(underscored_time)
make_csv_from_tx_traj(tx_traj, tx_data, fname)
