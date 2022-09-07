"""
Make a Coiled software environment for stackstac from the poetry lockfile's
current deps, for the [binder,viz] extras.

Pass --dev to build off the latest commit on `origin/main` instead.
"""

from pathlib import Path
import subprocess
import sys
import warnings

import coiled
import stackstac


def run(cmd: str, **kwargs) -> str:
    return subprocess.run(
        cmd, shell=True, check=True, capture_output=True, text=True, **kwargs
    ).stdout.strip()


if __name__ == "__main__":
    # TODO use click or something to make an actual CLI.
    dev = len(sys.argv) == 2 and sys.argv[-1] == "--dev"

    with (Path(__file__).parent.parent / "binder" / "runtime.txt").open() as f:
        line = f.read().strip()
        version = line.split("-")[1].split(".")
        version_str = ".".join(version[:2])
        print(f"Building for Python version: {version_str}")
        version = tuple(map(int, version))
        if version[:2] != sys.version_info[:2]:
            warnings.warn(
                f"senv version {version[:2]} does not match interpreter version {sys.version_info[:2]}"
            )

    deps = run("pdm export --without-hashes --format requirements -G binder -G viz").splitlines()

    if dev:
        subprocess.run("git fetch", shell=True, check=True)
        branch = run("git rev-parse --abbrev-ref HEAD")
        commit = run(f"git rev-parse {branch}")
        origin_commit = run(f"git rev-parse origin/{branch}")
        if commit != origin_commit:
            warnings.warn(
                f"Your local branch {branch!r} is not up to date with origin/{branch}"
            )
        name = f"stackstac-dev-{branch}"
        stackstac_dep = f"git+https://github.com/gjoseph92/stackstac.git@{origin_commit}#egg=stackstac[binder,viz]"
    else:
        name = "stackstac"
        version = stackstac.__version__
        stackstac_dep = f"stackstac[binder,viz]=={version}"

    deps += [stackstac_dep, "git+https://github.com/gjoseph92/scheduler-profilers.git@main"]
    print(f"Building senv {name!r} for {stackstac_dep}")

    coiled.create_software_environment(
        name=name,
        conda=[f"python={version_str}"],
        pip=deps,
        backend_options={"region": "us-west-2"},
    )
