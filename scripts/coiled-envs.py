"""
Make Coiled software environments for stackstac in multiple regions,
and upload all notebook files into a Coiled notebook (aka Job).

Pass --dev to build off the latest commit on `origin/main` instead.
"""

import asyncio
import io
import subprocess
from pathlib import Path
import sys
from typing import List
import warnings

import aiotools
import coiled


async def create_software_environment_quiet(
    cloud: coiled.Cloud[coiled.core.Async], **kwargs
) -> None:
    log = io.StringIO()
    try:
        await cloud.create_software_environment(log_output=log, **kwargs)
    except Exception:
        print(
            f"Error creating software environment with {kwargs}:\n{log.getvalue()}",
            file=sys.stderr,
        )
        raise

    print(f"Built environment for {kwargs.get('backend_options')}")


async def make_coiled_stuff(
    deps: List[str],
    regions: List[str],
    name: str,
) -> None:
    examples = Path(__file__).parent.parent / "examples"
    docs = Path(__file__).parent.parent / "docs"
    notebook_paths = list(examples.glob("*.ipynb")) + list(docs.glob("*.ipynb"))
    notebooks = list(map(str, notebook_paths))
    print(f"Notebooks: {notebooks}")

    async with coiled.Cloud(asynchronous=True) as cloud:
        name_software = name + "-notebook"
        async with aiotools.TaskGroup(name="envs") as tg:
            # Build all the software environments in parallel in multiple regions
            tg.create_task(
                create_software_environment_quiet(
                    cloud,
                    name=name_software,
                    container="coiled/notebook:latest",
                    pip=deps,
                )
            )

            for region in regions:
                tg.create_task(
                    create_software_environment_quiet(
                        cloud,
                        name=name,
                        pip=deps,
                        backend_options={"region": region},
                    )
                )

        # Create job configuration for notebook
        await cloud.create_job_configuration(
            name=name,
            software=name_software,
            cpu=2,
            gpu=0,
            memory="8 GiB",
            command=["/bin/bash", "start.sh", "jupyter", "lab"],
            files=notebooks,
            ports=[8888],
            description="Example notebooks from the stackstac documentation",
        )
        print(f"Created notebook {name}")


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

    asyncio.run(
        make_coiled_stuff(deps, regions=["us-west-2", "eu-central-1"], name=name)
    )
