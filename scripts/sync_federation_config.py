import subprocess
import sys
import tomllib
from pathlib import Path

CONFIG_PATH = Path(__file__).resolve().parent.parent / "federation.local.toml"


def main() -> None:
    if not CONFIG_PATH.exists():
        print(f"Config file not found: {CONFIG_PATH}", file=sys.stderr)
        sys.exit(1)

    with CONFIG_PATH.open("rb") as f:
        config = tomllib.load(f)

    sim = config.get("simulation", {})
    num_supernodes = sim.get("num-supernodes")
    num_cpus = sim.get("client-resources-num-cpus")
    num_gpus = sim.get("client-resources-num-gpus")

    if num_supernodes is None:
        print(
            "Key 'simulation.num-supernodes' not present in federation.local.toml",
            file=sys.stderr,
        )
        sys.exit(1)

    cmd = [
        "flwr",
        "federation",
        "simulation-config",
        f"--num-supernodes={num_supernodes}",
    ]
    if num_cpus is not None:
        cmd.append(f"--client-resources-num-cpus={num_cpus}")
    if num_gpus is not None:
        cmd.append(f"--client-resources-num-gpus={num_gpus}")

    print(f"exec:", " ".join(cmd))
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    main()