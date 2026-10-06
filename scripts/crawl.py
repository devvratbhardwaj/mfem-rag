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
import argparse
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

# Edit these lists and URLs here before fetching, as with the PDF sources above.
MFEM_CODE_REF = 'v4.10'
MFEM_EXAMPLE_DIRS = ('examples/petsc/',)
PETSC_HTML = 'https://petsc.org/release/'
PETSC_HTML_PATHS = [
    'manual/ksp/index.html',
    'manualpages/KSP/KSPSetTolerances/index.html',
    'manualpages/KSP/KSPSetFromOptions/index.html',
    'manualpages/KSP/KSPGMRES/index.html',
    'manualpages/PC/PCGAMG/index.html',
    'manualpages/PC/PCSetType/index.html',
]
SLEPC_HTML = 'https://slepc.upv.es/release/'
SLEPC_HTML_PATHS = [
    'documentation/manual/intro.html',
    'documentation/manual/eps.html',
    'documentation/manual/st.html',
    'manualpages/EPS/EPSSetFromOptions.html',
    'manualpages/EPS/EPSSetTolerances.html',
    'manualpages/EPS/EPSSetOperators.html',
    'manualpages/EPS/EPSSetProblemType.html',
    'manualpages/ST/STSetType.html',
]


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
    if not urls:
        raise ValueError('Refusing an empty download list.')
    config = "\n".join(
        f'url = "{url}"\noutput = "{destination / path}"'
        for path, url in sorted(urls.items())
    )
    subprocess.run(
        [
            "curl", "-sS", "--fail", "--retry", "2",
            '--location', '--connect-timeout', '20', '--max-time', '180',
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
    if tree.get('truncated'):
        raise ValueError('GitHub returned a truncated MFEM website tree.')
    paths = [
        entry["path"]
        for entry in tree["tree"]
        if entry["path"].startswith("src/") and entry["path"].endswith(".md")
    ]
    raw = f"https://raw.githubusercontent.com/{MFEM_WEB_REPO}/{tree['sha']}"
    download({path: f"{raw}/{path}" for path in paths}, destination)

    provenance(destination, f"repo: github.com/{MFEM_WEB_REPO}\nref: {MFEM_WEB_REF}\ntree: {tree['sha']}\n")
    print(f"    {tree['sha'][:12]}  {describe(destination)}", flush=True)


def fetch_html(name: str, base: str, paths: list[str]) -> None:
    print(f'==> {name}: {len(paths)} selected HTML pages', flush=True)
    destination = reset(name)
    download({path: base + path for path in paths}, destination)
    provenance(destination, f'source: {base}\npages: {len(paths)}\n')


def fetch_mfem_examples(name: str) -> None:
    print(f'==> {name}: MFEM integration examples @ {MFEM_CODE_REF}', flush=True)
    tree = json.loads(curl(f'https://api.github.com/repos/mfem/mfem/git/trees/{MFEM_CODE_REF}?recursive=1'))
    if tree.get('truncated'):
        raise ValueError('GitHub returned a truncated MFEM code tree.')
    paths = [entry['path'] for entry in tree['tree']
             if entry['path'].startswith(MFEM_EXAMPLE_DIRS)
             and entry['path'].endswith(('.cpp', '.hpp', '.opts'))]
    if not paths:
        raise ValueError('No MFEM integration examples matched MFEM_EXAMPLE_DIRS.')
    destination = reset(name)
    base = f'https://raw.githubusercontent.com/mfem/mfem/{MFEM_CODE_REF}/'
    download({path: base + path for path in paths}, destination)
    provenance(destination, f'repo: github.com/mfem/mfem\nref: {MFEM_CODE_REF}\ntree: {tree["sha"]}\n')


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
    choices = ['pdf', 'mfem-doxygen', 'mfem-web', 'mfem-examples', 'petsc-html', 'slepc-html']
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--source', choices=choices, action='append', help='fetch only these sources; repeatable')
    selected = parser.parse_args().source or choices
    for name in selected:
        if name == 'pdf':
            for filename, url in PDFS.items():
                fetch_pdf(filename, url)
        elif name == 'mfem-doxygen':
            fetch_mfem_doxygen(name)
        elif name == 'mfem-web':
            fetch_mfem_web(name)
        elif name == 'mfem-examples':
            fetch_mfem_examples(name)
        elif name == 'petsc-html':
            fetch_html(name, PETSC_HTML, PETSC_HTML_PATHS)
        elif name == 'slepc-html':
            fetch_html(name, SLEPC_HTML, SLEPC_HTML_PATHS)
    print(f"\ncorpus: {describe(RAW_DIR)} under {RAW_DIR}", flush=True)


if __name__ == "__main__":
    main()
