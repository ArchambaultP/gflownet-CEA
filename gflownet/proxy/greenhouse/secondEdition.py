import pickle
import os
import torch
from gflownet.proxy.base import Proxy
from data.greenhouse.secondEdition.extract import load_data
from botorch.models.transforms.input import InputStandardize
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from gpytorch.kernels import MaternKernel, ScaleKernel

class GreenHouseChallenge2ndEdition(Proxy):
    def __init__(self, model_fp="gflownet/proxy/greenhouse/secondEdition.pkl", n_samples=1000, **kwargs):
        super().__init__(**kwargs)
        train_X, train_Y, val_X, val_Y, test_X, test_Y = load_data()
        if os.path.exists(model_fp):
            print(f"Loading saved model: {model_fp}")
            with open(model_fp, 'rb') as file:
                self.model = pickle.load(file)
        else:
            print(f"Training new model: {model_fp}")
            self.model = self._train(train_X, train_Y)
        


        # self.n_samples = n_samples
        # samples = self.model.posterior.rsample(sample_shape=torch.Size([self.n_samples]))
        # samples = samples.squeeze(-1).cpu().numpy()


    # def set_n_dim(self, n_dim):
    #     # self.n_dim is never used in this env,
    #     # this is just to make molecule env work with htorus
    #     self.n_dim = n_dim

    # Decorator to stop proxy model from training
    # @torch.no_grad()
    def __call__(self, states_proxy):
        # output of the model is exp(-energy) / 100
        # x = states_proxy % (2 * np.pi)
        # rewards = -np.log(self.model.predict(x) * 100)
        # return torch.tensor(
        #     rewards,
        #     dtype=self.float,
        #     device=self.device,
        # )
        
        # breakpoint()

        print("test")

        # TODO: set reward as distance between sampled GP and states_proxy
        return torch.tensor(100,
                            dtype=self.float,
                            device=self.device)
        

    def __deepcopy__(self, memo):
        cls = self.__class__
        new_obj = cls.__new__(cls)
        new_obj.__dict__.update(self.__dict__)
        return new_obj

    def _train(self, train_X, train_Y):
        in_standard = InputStandardize(d=train_X.shape[-1]).to(train_X)
        out_standard = Standardize(m=train_Y.shape[-1]).to(train_Y)
        model = SingleTaskGP(train_X,
                            train_Y,
                            input_transform=in_standard,
                            outcome_transform=out_standard,
                            covar_module=ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=train_X.shape[-1])))
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        mll = fit_gpytorch_mll(mll)
        return model