"""Download the exact HAI files used in the experiments and verify SHA-256."""

from __future__ import annotations

import argparse
import hashlib
import time
import urllib.request
from pathlib import Path

OLD = "https://raw.githubusercontent.com/icsdataset/hai/master/hai-21.03"
NEW = "https://media.githubusercontent.com/media/icsdataset/hai/master/hai-23.05"
HASHES = {
    "train1.csv.gz": "fdd0162315b32817ba665e2cc8990eb09c47c4edc29683a86b11dc9df062b3a9",
    "train2.csv.gz": "c330af06a24f5fe3df864dddc0e9cbbbfd27db2e005bd32dc8ab3eb2ab356337",
    "train3.csv.gz": "de09120c617bad5f4f3cab86705137fe7ced4b1123fb67ffdf67fd645f3d1b8f",
    "test1.csv.gz": "532de41bed372206941609d05b763958f59abecfd63c8c123649f823452e445e",
    "test2.csv.gz": "c5572e0ed19906107ceb56c4b4f40e165153fc7ba87b3506230d53d2478c29bb",
    "test3.csv.gz": "bff835894baff0c73871bc01ece4f5cf759d455a049e808b012128767420caaf",
    "test4.csv.gz": "51380fd28fd985b94a10cae3d8b80926becb3af9f233f2d3a38ab8be9482a482",
    "test5.csv.gz": "f0d59308b167ed95d0e81fb3e27c906682c9b21d3dba0ab4c4d696565df35179",
}
NEW_HASHES = {
    "hai-test1.csv": "78c7f1d4de1f2ab9ccc2f8c719f80f831033543adb0c81d0d78f84f40838d4be",
    "hai-test2.csv": "b2b8dd295aefd87e39260fe43cb4c73ee86d6264b0ac4b0761e7efb0c2b545c3",
    "label-test1.csv": "eaf69edb9c5834bc393afd7bf658b5e408d34fd7bfc3261f80516765fb818fbc",
    "label-test2.csv": "8090c44981176e39b0f01a7126a80248ac0b93355c00f9db4d4e2f2106452b92",
    "summary_label1.txt": "9d8fb32ee1e816c88f93c8d661f90a7736802582d3572a1e1ebd953783353da6",
    "summary_label2.txt": "b564024a105981415a81dd1b338a48ad7ad5be511849884232540f1a6550e5c1",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _download(url: str, output: Path, expected: str, retries: int = 3) -> None:
    if output.exists() and _sha256(output) == expected:
        print(f"verified {output}")
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".part")
    for attempt in range(retries):
        try:
            digest = hashlib.sha256()
            with urllib.request.urlopen(url, timeout=60) as response, temporary.open("wb") as handle:
                for block in iter(lambda: response.read(1024 * 1024), b""):
                    digest.update(block)
                    handle.write(block)
            if digest.hexdigest() != expected:
                raise OSError(f"SHA-256 mismatch for {url}")
            temporary.replace(output)
            print(f"downloaded {output}")
            return
        except Exception:
            temporary.unlink(missing_ok=True)
            if attempt + 1 == retries:
                raise
            time.sleep(2**attempt)


def download(root: Path) -> None:
    for name, digest in HASHES.items():
        _download(f"{OLD}/{name}", root / "hai-21.03" / name, digest)
    for name, digest in NEW_HASHES.items():
        _download(f"{NEW}/{name}", root / "hai-23.05" / name, digest)
    _download(
        "https://raw.githubusercontent.com/icsdataset/hai/master/hai_dataset_technical_details.pdf",
        root / "hai_dataset_technical_details.pdf",
        "0668345c4e80331b918fe17c81f8f363b13bd22886831d286e761bc62b71a556",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/raw"))
    args = parser.parse_args()
    download(args.root)


if __name__ == "__main__":
    main()
