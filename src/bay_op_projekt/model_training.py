from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.models.model_list_gp_regression import ModelListGP
import torch

def train_gp_model(train_X, train_Y, train_Yvar, use_pareto_front:bool=False):
    # GP-Modell erstellen

    if use_pareto_front:
        model = train_with_pareto_front(train_X, train_Y, train_Yvar)

        # For ModelListGP, optimize each sub-model separately
        for m in model.models:
            mll = ExactMarginalLogLikelihood(m.likelihood, m)
            fit_gpytorch_mll(mll)

        model.eval()
    else:
        model = SingleTaskGP(
            train_X=train_X,
            train_Y=train_Y,
            train_Yvar=train_Yvar
        )

        # Hyperparameter des GP optimieren
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)
        model.eval()

    return model

def train_with_pareto_front(train_X, train_Y, train_Yvar) -> ModelListGP:

    if train_Y.dim() == 1:
        train_Y = train_Y.unsqueeze(-1)
    if train_Yvar.dim() == 1:
        train_Yvar = train_Yvar.unsqueeze(-1)

    train_Y1 = torch.tensor(train_Y, dtype=torch.double)
    train_Y2 = torch.tensor(-train_Yvar, dtype=torch.double)

    model1 = SingleTaskGP(train_X, train_Y1)
    model2 = SingleTaskGP(train_X, train_Y2)

    model  = ModelListGP(model1, model2)

    return model