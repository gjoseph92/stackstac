"""
Make Coiled software environments for stackstac in multiple regions,
and upload all notebook files into a Coiled notebook (aka Job).
"""

import asyncio
import io
import subprocess
from pathlib import Path
import sys
from typing import List

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


if __name__ == "__main__":
    proc = subprocess.run(
        "poetry export --without-hashes -E binder -E viz",
        shell=True,
        check=True,
        capture_output=True,
        text=True,
    )
    deps = proc.stdout.splitlines()

    # TODO single-source the version! this is annoying
    version = subprocess.run(
        "poetry version -s", shell=True, check=True, capture_output=True, text=True
    ).stdout.strip()
    print(f"Version: {version}")
    deps += [f"stackstac[binder,viz]=={version}"]

    asyncio.run(
        make_coiled_stuff(deps, regions=["us-west-2", "eu-central-1"], name="stackstac")
    )
