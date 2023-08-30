class MontecarloMethod():
    def __init__(self, montecarlo_method):
        self.montecarlo_method = montecarlo_method

    def __call__(self, montecarlo_outputs):
        if self.montecarlo_method == "sum":
            montecarlo_outputs = torch.sum(montecarlo_outputs, dim=0)
        elif self.montecarlo_method == "mean":
            montecarlo_outputs = torch.mean(montecarlo_outputs, dim=0)
        else:
            raise Exception("Invalid montecarlo method")
