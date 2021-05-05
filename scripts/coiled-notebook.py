import subprocess

from pathlib import Path
import coiled

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
    deps += [f"stackstac[binder,viz]=={version}"]

    examples = Path(__file__).parent.parent / "examples"
    docs = Path(__file__).parent.parent / "docs"
    notebooks = list(examples.glob("*.ipynb")) + list(docs.glob("*.ipynb"))
    print(notebooks)
    coiled.create_notebook(
        name="stackstac",
        pip=deps,
        cpu=2,
        memory="8 GiB",
        files=notebooks,
        description="Example notebooks from the stackstac documentation (https://stackstac.readthedocs.io/en/stable/)",
    )
