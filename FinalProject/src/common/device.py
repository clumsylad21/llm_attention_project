import torch


def choose_device(device_arg: str) -> str:
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_arg


def choose_dtype(dtype_arg: str, device: str) -> torch.dtype:
    mapping = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }

    dtype = mapping[dtype_arg]

    # CPU baseline ke liye fp32 safest hai
    if device == "cpu" and dtype != torch.float32:
        print("[warn] CPU selected; switching dtype to fp32 for compatibility.")
        return torch.float32

    return dtype


def sync_if_needed(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


def get_device_name(device: str) -> str:
    if device == "cuda":
        return torch.cuda.get_device_name(0)
    return "CPU"