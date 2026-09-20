"""Module 1 - Documentation crawler

Downloads the PETSc and SLEPc manuals into data/raw/pdf/. 
Runs unconditionally: no cache check, no version pinning.

MFEM publishes no PDF manual, so its documentation needs a site crawl. 
That is deferred for now.

"""

import subprocess
from pathlib import Path

RAW_DIR = Path("../data/raw")

# filename -> url
PDFS = {
    "petsc-manual.pdf": "https://petsc.org/release/manual/manual.pdf",
    "slepc-manual.pdf": "https://slepc.upv.es/release/documentation/manual/slepc-manual.pdf",
}


def fetch_pdf(filename: str, url: str) -> None:
    print(f"==> fetching {filename}: {url}", flush=True)
    destination = RAW_DIR / "pdf" / filename
    destination.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["curl", "-fsSL", "--retry", "3", "-o", str(destination), url], check=True)


def main() -> None:
    for filename, url in PDFS.items():
        fetch_pdf(filename, url)


if __name__ == "__main__":
    main()
