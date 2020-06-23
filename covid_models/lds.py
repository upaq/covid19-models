import numpy as np
import pybasicbayes
import pylds.models

from pybasicbayes.util.stats import sample_mniw
from typing import Tuple


def VanillaLDS(D_obs, D_latent, D_input=0,
               mu_init=None, sigma_init=None,
               A=None, B=None, sigma_states=None,
               C=None, D=None, sigma_obs=None):
    model = pylds.models.LDS(
        dynamics_distn=pybasicbayes.distributions.Regression(
            nu_0=D_latent + 1,
            S_0=D_latent * np.eye(D_latent),
            M_0=np.zeros((D_latent, D_latent + D_input)),
            K_0=D_latent * np.eye(D_latent + D_input)),
        emission_distn=pybasicbayes.distributions.Regression(
            nu_0=D_obs + 1,
            S_0=D_obs * np.eye(D_obs),
            M_0=np.zeros((D_obs, D_latent + D_input)),
            K_0=D_obs * np.eye(D_latent + D_input)))

    set_default = \
        lambda prm, val, default: \
            model.__setattr__(prm, val if val is not None else default)

    set_default('mu_init', mu_init, np.zeros(D_latent))
    set_default('sigma_init', sigma_init, np.eye(D_latent))

    set_default('A', A, 0.99 * pylds.util.random_rotation(D_latent))
    set_default('B', B, 0.1 * np.random.randn(D_latent, D_input))
    set_default('sigma_states', sigma_states, 0.1 * np.eye(D_latent))

    set_default('C', C, np.random.randn(D_obs, D_latent))
    set_default('D', D, 0.1 * np.random.randn(D_obs, D_input))
    set_default('sigma_obs', sigma_obs, 0.1 * np.eye(D_obs))

    return model


class LDS(object):
    def __init__(self, D_obs: int, D_latent: int, D_input: int,
                 D_input_gamma: int):
        assert D_input_gamma <= D_input

        self.D_obs = D_obs
        self.D_latent = D_latent
        self.D_input = D_input
        self.D_input_gamma = D_input_gamma

        self.C_constant = np.zeros((self.D_obs, self.D_latent))
        self.C_constant[:, -1] = 1

        # We let gamma ~ N(1, 1) distribution. It scales a subset of the
        # external controls (e.g. the Oxford government stringency index)
        # per country or time series. A mean at one (and not zero) breaks
        # symmetry; we could alternatively place a prior on it to enforce
        # non-negativity.
        self.gamma_prior_precision = 1
        self.gamma_prior_mean_times_precision = 1

        mu_init = np.zeros(D_latent)
        sigma_init = np.eye(D_latent)

        self.model = VanillaLDS(self.D_obs, self.D_latent, self.D_input,
                                A=np.eye(self.D_latent),
                                B=np.zeros((self.D_latent, self.D_input)),
                                C=np.ones((self.D_obs, self.D_latent)),
                                D=np.zeros((self.D_obs, self.D_input)),
                                mu_init=mu_init,
                                sigma_init=sigma_init)
        self.gammas = []
        self.inputs = []

    def add_data(self, x, inputs):
        self.model.add_data(x, inputs=inputs)
        self.gammas.append(1.0)
        self.inputs.append(inputs)

    def _resample_dynamics_identity_control(self, data):

        def _empty_statistics(D_in, D_out):
            return np.array([np.zeros((D_out, D_out)),
                             np.zeros((D_out, D_in)),
                             np.zeros((D_in, D_in)), 0])

        stats = sum((self.model.dynamics_distn._get_statistics(d)
                     for d in data),
                    _empty_statistics(self.D_input, self.D_latent))

        # TODO. The prior parameters should be taken from the model,
        # not restated here.
        natural_hypparam = self.model.dynamics_distn._standard_to_natural(
            self.D_latent + 1,  # nu_0
            self.D_latent * np.eye(self.D_latent),  # S_0
            self.np.zeros((self.D_latent, self.D_input)),  # M_0
            self.D_latent * np.eye(self.D_input)  # K_0
        )
        mean_params = self.model.dynamics_distn._natural_to_standard(
            natural_hypparam + stats)

        A, sigma = sample_mniw(*mean_params)

        self.model.dynamics_distn.sigma = sigma
        self.model.dynamics_distn.A = np.concatenate(
            (np.eye(self.D_latent), A), axis=1)
        self.model.dynamics_distn._initialize_mean_field()

    def _resample_gamma(self):
        for i, (s, inputs) in enumerate(zip(self.model.states_list,
                                            self.inputs)):
            inputs = inputs[:-1]

            # We break the inputs into [inputs1, inputs2], where inputs1
            # correspond to the dimensions that are rescaled by gamma.
            # -- Precision --
            inputs1 = inputs[:, :self.D_input_gamma]
            B1 = self.model.B[:, :self.D_input_gamma]
            i1 = np.matmul(B1, inputs1.T)

            y = np.matmul(np.linalg.inv(self.model.sigma_states), i1)
            precision = np.sum(y * i1) + self.gamma_prior_precision

            # -- Mean times precision --
            inputs2 = inputs[:, self.D_input_gamma:]
            B2 = self.model.B[:, self.D_input_gamma:]
            i2 = np.matmul(B2, inputs2.T)

            Az = np.matmul(self.model.A, s.gaussian_states[:-1].T)
            y = s.gaussian_states[1:].T - Az - i2
            y = np.matmul(np.linalg.inv(self.model.sigma_states), y)
            mean_times_precision = np.sum(
                y * i1) + self.gamma_prior_mean_times_precision

            mu = mean_times_precision / precision
            sigma = 1 / np.sqrt(precision)
            gamma = np.random.normal(loc=mu, scale=sigma)

            self.gammas[i] = gamma

    def resample_model(self,
                       identity_transition_matrix: bool=False,
                       fixed_emission_matrix: bool=False,
                       include_gamma: bool=False) -> Tuple['MCMCSample',
                                                           float]:

        # 1.  Resample the parameters
        # 1.1 Resample the scalar gamma for each time series' external control
        if include_gamma:
            self._resample_gamma()
            self._scale_inputs_by_gamma()

        # 1.2 Resample the dynamics distribution
        if identity_transition_matrix:
            data = [np.hstack((s.inputs[:-1],
                               s.gaussian_states[1:] - s.gaussian_states[:-1])
                              )
                    for s in self.model.states_list]

            self._resample_dynamics_identity_control(data)
        else:
            self.model.resample_dynamics_distn()

        # 1.3 Resample the emission distribution
        xys = [(np.hstack((s.gaussian_states, 0 * s.inputs)),
                s.data)
               for s in self.model.states_list]

        self.model.emission_distn.resample(data=xys)

        self.model.D = np.zeros((self.D_obs, self.D_input))

        if fixed_emission_matrix:
            # Comment. In pybasicbayes's sampler for Multivariate Normal
            # Inverse Wishart, sample_mniw(...), Sigma is first sampled,
            # and then A conditioned on Sigma. It is therefore safe to reset
            #  C to C_constant. We therefore just keep the emission noise
            # sample.
            self.model.C = self.C_constant

        # 2.  Resample the states
        self.model.resample_states()

        sample = MCMCSample(A=self.model.A, B=self.model.B,
                            C=self.model.C, D=self.model.D,
                            Q=self.model.dynamics_distn.sigma,
                            R=self.model.emission_distn.sigma,
                            gamma=np.array(self.gammas))

        return sample, self.model.log_likelihood()

    # Assuming the initial parameters are decent, first sample states.
    def resample_states(self):
        self.model.resample_states()

    def _scale_inputs_by_gamma(self):
        for s, inputs, gamma in zip(self.model.states_list,
                                    self.inputs, self.gammas):
            gamma_mask = np.concatenate(
                (
                    gamma * np.ones((1, self.D_input_gamma)),
                    np.ones((1, self.D_input - self.D_input_gamma))
                ),
                axis=1)

            s.inputs = inputs * gamma_mask

            # should other statistics be set?


class MCMCSample(object):
    def __init__(self, A, B, C, D, Q, R, gamma):
        self.A = A
        self.B = B
        self.C = C
        self.D = D
        self.Q = Q
        self.R = R
        self.gamma = gamma

    def __str__(self):
        string = 'A:\n' + str(self.A) + '\nB:\n' + str(
            self.B) + '\nC:\n' + str(self.C) + '\nD:\n' + str(
            self.D) + '\nQ:\n' + str(self.Q) + '\nR:\n' + str(
            self.R) + '\ngamma:\n' + str(self.gamma)
        return string
