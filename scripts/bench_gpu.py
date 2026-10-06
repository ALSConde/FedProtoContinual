import time, torch
x = torch.randn(32, 64, device="cuda")

def bench(n, sync):
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(n):
        y = (x @ x.T).sum()
        if sync:
            y.item()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / n * 1e3

bench(50, True)
print("ms por op, sem sync:", bench(2000, False), "| com sync:", bench(2000, True))