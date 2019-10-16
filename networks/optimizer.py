from torch import optim

from networks.optimizers.radam import RAdam

def get(name):
    optimizer_fn = dict(
        Adam = optim.Adam,
        RAdam = RAdam
    )

    return optimizer_fn[name]
