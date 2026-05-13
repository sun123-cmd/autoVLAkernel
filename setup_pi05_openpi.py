"""
One-time setup for profiling openpi's pi0.5 model from the AutoKernel repo.

This script is meant to be run from the autokernel directory via:

    uv run setup_pi05_openpi.py

It installs openpi into the currently active AutoKernel uv environment, applies
the required transformers patches, and can optionally download + convert the
pi05_droid checkpoint to PyTorch format.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path
import subprocess
import shutil
import sys
import glob


CHECKPOINT_URI = "gs://openpi-assets/checkpoints/pi05_droid"
CONFIG_NAME = "pi05_droid"


def _repo_dirs() -> tuple[Path, Path]:
    autokernel_dir = Path(__file__).resolve().parent
    openpi_dir = autokernel_dir.parent / "openpi"
    if not openpi_dir.exists():
        raise SystemExit(f"Sibling openpi checkout not found: {openpi_dir}")
    return autokernel_dir, openpi_dir


def _uv_bin() -> str:
    return shutil.which("uv") or "/home/sunwenhao/.local/bin/uv"


def _run(cmd: list[str], *, cwd: Path, env: dict[str, str] | None = None) -> None:
    print(f"+ (cd {cwd} && {' '.join(cmd)})")
    subprocess.run(cmd, cwd=cwd, env=env, check=True)


def _current_python_minor() -> tuple[int, int]:
    return sys.version_info.major, sys.version_info.minor


def _missing_modules(names: list[str]) -> list[str]:
    return [name for name in names if importlib.util.find_spec(name) is None]


def _transformers_dir() -> Path:
    spec = importlib.util.find_spec("transformers")
    if spec is None or not spec.submodule_search_locations:
        raise SystemExit(
            "transformers is not importable from the current autokernel uv environment.\n"
            "Retry with:\n"
            "  uv sync\n"
            "  uv run setup_pi05_openpi.py --download --convert"
        )
    return Path(next(iter(spec.submodule_search_locations))).resolve()


def _has_nccl_symbol(lib_path: Path, symbol: str = "ncclCommWindowDeregister") -> bool:
    if not lib_path.exists():
        return False
    result = subprocess.run(["nm", "-D", str(lib_path)], capture_output=True, text=True, check=False)
    return symbol in result.stdout


def _find_good_nccl_library() -> Path | None:
    candidate_patterns = [
        str(Path("~/.cache/uv/archive-v0/*/nvidia/nccl/lib/libnccl.so.2").expanduser()),
        str(Path("~/miniconda3/**/site-packages/nvidia/nccl/lib/libnccl.so.2").expanduser()),
        "/usr/**/libnccl.so.2",
        "/opt/**/libnccl.so.2",
    ]
    seen: set[Path] = set()
    for pattern in candidate_patterns:
        for match in glob.glob(pattern, recursive=True):
            path = Path(match).resolve()
            if path in seen:
                continue
            seen.add(path)
            if _has_nccl_symbol(path):
                return path
    return None


def _ensure_working_nccl() -> None:
    site_packages = Path(next(p for p in sys.path if p.endswith("site-packages")))
    target = site_packages / "nvidia" / "nccl" / "lib" / "libnccl.so.2"
    if _has_nccl_symbol(target):
        return

    candidate = _find_good_nccl_library()
    if candidate is None:
        raise SystemExit(
            "Could not find a working libnccl.so.2 with ncclCommWindowDeregister on this machine."
        )

    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(candidate, target)
    print(f"Repaired NCCL runtime: {candidate} -> {target}")


def _default_converted_dir() -> Path:
    return Path("~/.cache/openpi/openpi-assets/checkpoints/pi05_droid_pytorch").expanduser()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download the official pi05_droid JAX checkpoint into ~/.cache/openpi.",
    )
    parser.add_argument(
        "--convert",
        action="store_true",
        help="Convert the downloaded pi05_droid JAX checkpoint to a PyTorch checkpoint.",
    )
    parser.add_argument(
        "--checkpoint-uri",
        default=CHECKPOINT_URI,
        help=f"Checkpoint URI to download (default: {CHECKPOINT_URI}).",
    )
    parser.add_argument(
        "--converted-dir",
        default=str(_default_converted_dir()),
        help="Directory for the converted PyTorch checkpoint.",
    )
    args = parser.parse_args()

    if _current_python_minor() < (3, 11):
        raise SystemExit(
            "pi0.5 via openpi requires Python 3.11+.\n"
            "From /home/sunwenhao/VLAinfra/autokernel run:\n"
            "  uv python pin 3.11\n"
            "  uv sync\n"
            "Then rerun:\n"
            "  uv run setup_pi05_openpi.py"
        )

    virtual_env = os.environ.get("VIRTUAL_ENV")
    if not virtual_env:
        raise SystemExit(
            "No active uv-managed environment detected.\n"
            "Run this script via:\n"
            "  uv run setup_pi05_openpi.py"
        )

    _, openpi_dir = _repo_dirs()
    env = os.environ.copy()
    env["GIT_LFS_SKIP_SMUDGE"] = "1"
    uv_bin = _uv_bin()

    _run([uv_bin, "sync", "--active"], cwd=openpi_dir, env=env)
    _run(
        [uv_bin, "pip", "install", "--python", sys.executable, "-e", "."],
        cwd=openpi_dir,
        env=env,
    )
    _ensure_working_nccl()

    missing = _missing_modules(["openpi", "transformers", "jax", "flax", "sentencepiece"])
    if missing:
        raise SystemExit(
            "openpi dependencies are still missing from the current autokernel uv environment: "
            + ", ".join(missing)
            + "\n"
            + "Retry from /home/sunwenhao/VLAinfra/autokernel with:\n"
            + "  uv sync\n"
            + "  uv run setup_pi05_openpi.py --download --convert"
        )

    _run(
        [
            sys.executable,
            "scripts/patch_transformers_replace.py",
            "--transformers-dir",
            str(_transformers_dir()),
        ],
        cwd=openpi_dir,
        env=env,
    )

    if args.download or args.convert:
        _run(
            [
                sys.executable,
                "-c",
                (
                    "from openpi.shared import download; "
                    f"print(download.maybe_download('{args.checkpoint_uri}'))"
                ),
            ],
            cwd=openpi_dir,
            env=env,
        )

    if args.convert:
        checkpoint_name = args.checkpoint_uri.rstrip("/").split("/")[-1]
        checkpoint_dir = Path("~/.cache/openpi/openpi-assets/checkpoints").expanduser() / checkpoint_name
        _run(
            [
                sys.executable,
                "examples/convert_jax_model_to_pytorch.py",
                "--checkpoint_dir",
                str(checkpoint_dir),
                "--config_name",
                CONFIG_NAME,
                "--output_path",
                str(Path(args.converted_dir).expanduser()),
            ],
            cwd=openpi_dir,
            env=env,
        )

    converted_dir = Path(args.converted_dir).expanduser()
    print()
    print("pi0.5 setup complete.")
    print()
    print("Next commands:")
    print("  export OPENPI_PI05_PT_CHECKPOINT=" + str(converted_dir))
    print("  uv run prepare.py")
    print("  uv run profile.py --model models/pi05_openpi.py --class-name PI05AutoKernelModel \\")
    print("   --input-shape 1,10 --dtype bfloat16")
    print("  uv run extract.py --top 5")
    print("  uv run bench.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
