import torch


def transform_to_tensor(x, dtype=torch.float, grad=True, device="cuda"):
    if isinstance(x, dict):
        tensor = {
            k: torch.tensor(v, dtype=dtype, device=device, requires_grad=grad)
            for k, v in x.items()
        }
    else:
        tensor = torch.tensor(
            x, dtype=dtype, device=device, requires_grad=grad
        )  # B, S_D
    return tensor
