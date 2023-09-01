import torch


class MontecarloMethod():
    def __init__(self, montecarlo_method):
        self.montecarlo_method = montecarlo_method

    def __call__(self, montecarlo_outputs):

        if self.montecarlo_method == "sum":
            montecarlo_outputs = torch.sum(montecarlo_outputs, dim=1)
        elif self.montecarlo_method == "mean":
            montecarlo_outputs = torch.mean(montecarlo_outputs, dim=1)
        elif self.montecarlo_method == "std":
            montecarlo_outputs = torch.std(montecarlo_outputs, dim=1)
        elif self.montecarlo_method == "norm2":
            montecarlo_outputs = torch.norm(montecarlo_outputs, p=2, dim=1)
        else:
            raise Exception("Invalid montecarlo method")
        return montecarlo_outputs.item()
