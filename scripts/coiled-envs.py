"""
Make a Coiled software environment for stackstac from the poetry lockfile's
current deps, for the [binder,viz] extras.

Pass --dev to build off the latest commit on `origin/main` instead.
"""

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
        print(f"Version: {version}")
        stackstac_dep = f"stackstac[binder,viz]=={version}"
    deps += [stackstac_dep]

    coiled.create_software_environment(
        name=name,
        conda=["python=3.8"],
        pip=deps,
        backend_options={"region": "us-west-2"},
    )
