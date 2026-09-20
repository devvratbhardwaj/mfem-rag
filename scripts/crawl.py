"""Module 1 - documentation crawler

Puts the MFEM, PETSc and SLEPc documentation on local disk under data/raw/.

Everything is fetched over HTTP with curl. MFEM publishes no PDF manual; its
documentation is the Doxygen API reference at docs.mfem.org plus the narrative
site at mfem.org, so those arrive as many small files rather than one big one.
Neither needs a git clone - both expose a manifest listing their own pages:

- docs.mfem.org ships doxygen_crawl.html, Doxygen's own index of every page.
- mfem/web's file list comes from the GitHub tree API, which also returns the
  tree sha, so the fetch stays pinned without cloning anything.

Runs unconditionally: no cache check, no staleness window. A source directory
that already exists is removed and re-fetched.
"""

import json
import re
import shutil
import subprocess
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"

# filename -> url
PDFS = {
    "petsc-manual.pdf": "https://petsc.org/release/manual/manual.pdf",
    "slepc-manual.pdf": "https://slepc.upv.es/release/documentation/manual/slepc-manual.pdf",
}

# Doxygen publishes one frozen directory per release. The version in the URL is
# the version pin: /4.10/ does not change once released.
MFEM_DOXYGEN = "https://docs.mfem.org/4.10"

MFEM_WEB_REPO = "mfem/web"
MFEM_WEB_REF = "master"


def keep_doxygen_page(name: str) -> bool:
    """Class, struct and namespace reference pages: the documented API surface.

    Everything else on the site is graphics (112 MB of SVG call graphs), source
    listings (144 MB of *_source.html), or navigation. Two exclusions are worth
    naming because they look like content and are not:

    - *-members.html   flat "all members including inherited" tables, a
                       near-duplicate of the class page beside them
    - namespacemembers_*.html
                       alphabetical member indexes, the namespace counterpart
                       of the functions_*.html pages
    """
    if not name.endswith(".html") or not name.startswith(
        ("class", "struct", "namespace")
    ):
        return False
    return not name.endswith("-members.html") and not name.startswith(
        "namespacemembers_"
    )


def curl(url: str) -> str:
    return subprocess.run(
        ["curl", "-fsSL", "--retry", "3", url],
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    ).stdout


def download(urls: dict[str, str], destination: Path) -> None:
    """Fetch `urls` (local relative path -> url) into `destination`.

    One curl invocation driven by a config file on stdin, so 1,600 pages cost
    one process and a shared connection pool rather than 1,600 of each.
    --fail makes any 404 abort the run: a silently short corpus is worse than a
    crash.
    """
    config = "\n".join(
        f'url = "{url}"\noutput = "{destination / path}"'
        for path, url in sorted(urls.items())
    )
    subprocess.run(
        [
            "curl", "-sS", "--fail", "--retry", "2",
            "--create-dirs",
            "--parallel", "--parallel-max", "16",
            "--config", "-",
        ],
        input=config,
        text=True,
        check=True,
    )


def fetch_pdf(filename: str, url: str) -> None:
    print(f"==> {filename}: {url}", flush=True)
    destination = RAW_DIR / "pdf" / filename
    destination.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["curl", "-fsSL", "--retry", "3", "-o", str(destination), url], check=True
    )
    print(f"    {destination.stat().st_size / 1e6:.1f} MB", flush=True)


def fetch_mfem_doxygen(name: str) -> None:
    """MFEM API reference, selected from Doxygen's own page manifest."""
    print(f"==> {name}: MFEM API reference (Doxygen)", flush=True)
    print(f"    {MFEM_DOXYGEN}", flush=True)
    destination = reset(name)

    listed = set(re.findall(r'href="([^"]+)"', curl(f"{MFEM_DOXYGEN}/doxygen_crawl.html")))
    pages = sorted(page for page in listed if keep_doxygen_page(page))
    download({page: f"{MFEM_DOXYGEN}/{page}" for page in pages}, destination)

    provenance(destination, f"source: {MFEM_DOXYGEN}\npages: {len(pages)} of {len(listed)} listed\n")
    print(f"    {describe(destination)}", flush=True)


def fetch_mfem_web(name: str) -> None:
    """MkDocs sources behind mfem.org - the narrative layer.

    Taken as Markdown rather than as rendered mfem.org HTML: the headings give
    the chunker a structure to split on, and there is no navigation chrome to
    strip back off.
    """
    print(f"==> {name}: MFEM narrative documentation (mfem.org sources)", flush=True)
    print(f"    github.com/{MFEM_WEB_REPO} @ {MFEM_WEB_REF}", flush=True)
    destination = reset(name)

    tree = json.loads(
        curl(
            f"https://api.github.com/repos/{MFEM_WEB_REPO}/git/trees/{MFEM_WEB_REF}?recursive=1"
        )
    )
    paths = [
        entry["path"]
        for entry in tree["tree"]
        if entry["path"].startswith("src/") and entry["path"].endswith(".md")
    ]
    raw = f"https://raw.githubusercontent.com/{MFEM_WEB_REPO}/{tree['sha']}"
    download({path: f"{raw}/{path}" for path in paths}, destination)

    provenance(destination, f"repo: github.com/{MFEM_WEB_REPO}\nref: {MFEM_WEB_REF}\ntree: {tree['sha']}\n")
    print(f"    {tree['sha'][:12]}  {describe(destination)}", flush=True)


def reset(name: str) -> Path:
    destination = RAW_DIR / name
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    return destination


def provenance(destination: Path, body: str) -> None:
    (destination / "SOURCE.txt").write_text(f"{body}fetched: {date.today()}\n")


def describe(directory: Path) -> str:
    files = [p for p in directory.rglob("*") if p.is_file()]
    return f"{len(files)} files, {sum(p.stat().st_size for p in files) / 1e6:.1f} MB"


def main() -> None:
    for filename, url in PDFS.items():
        fetch_pdf(filename, url)
    fetch_mfem_doxygen("mfem-doxygen")
    fetch_mfem_web("mfem-web")
    print(f"\ncorpus: {describe(RAW_DIR)} under {RAW_DIR}", flush=True)


if __name__ == "__main__":
    main()
