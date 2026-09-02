#!/usr/bin/env python3

import os
import socket
import time

import torch


def main():
    print(f"hostname={socket.gethostname()}", flush=True)
    print(f"slurm_job_id={os.environ.get('SLURM_JOB_ID', '')}", flush=True)
    print(f"torch={torch.__version__}", flush=True)
    print(f"torch_cuda={torch.version.cuda}", flush=True)
    print(f"cuda_available={torch.cuda.is_available()}", flush=True)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available inside the Slurm job.")

    device = torch.device("cuda:0")
    print(f"gpu_count={torch.cuda.device_count()}", flush=True)
    print(f"gpu_name={torch.cuda.get_device_name(device)}", flush=True)

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    a = torch.randn(4096, 4096, device=device)
    b = torch.randn(4096, 4096, device=device)

    torch.cuda.synchronize()
    started = time.perf_counter()
    c = a @ b
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started

    print(f"matmul_seconds={elapsed:.6f}", flush=True)
    print(f"checksum={c.float().mean().item():.8f}", flush=True)
    print("GPU_SMOKE_TEST=PASS", flush=True)


if __name__ == "__main__":
    main()
