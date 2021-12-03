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
        if version[:2] != sys.version_info[:2]:
            warnings.warn(
                f"senv version {version_str} does not match interpreter version {sys.version_info[:2]}"
            )

    deps = run("poetry export --without-hashes -E binder -E viz").splitlines()

    if dev:
        name = "stackstac-dev"
        subprocess.run("git fetch", shell=True, check=True)
        main = run("git rev-parse main")
        origin_main = run("git rev-parse origin/main")
        if main != origin_main:
            warnings.warn("Your local main branch is not up to date with origin/main")
        print(f"Commit (origin/main): {origin_main}")
        stackstac_dep = f"git+https://github.com/gjoseph92/stackstac.git@{origin_main}#egg=stackstac[binder,viz]"
    else:
        name = "stackstac"
        # TODO single-source the version! this is annoying
        version = run("poetry version -s")
        print(f"stackstac version: {version}")
        stackstac_dep = f"stackstac[binder,viz]=={version}"
    deps += [stackstac_dep]

    coiled.create_software_environment(
        name=name,
        conda=[f"python={version_str}"],
        pip=deps,
        backend_options={"region": "us-west-2"},
    )
