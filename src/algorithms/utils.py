import torch.nn


def save_model(model : torch.nn.Module, path : str, index : int, index_upper_bound : int, name : str):
    width = len(str(index_upper_bound))

    model.eval()

    scripted_model = torch.jit.script(model)

    scripted_model.save(f'{path}/{name}_{index:0{width}d}.pt')