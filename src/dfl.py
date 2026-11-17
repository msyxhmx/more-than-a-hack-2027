"""Training data for the tree model: the 7 open DFL matches (Bassek et al. 2025, CC-BY 4.0).

    python src/dfl.py          # download (if missing), check, convert -> data/external/dfl_npz

These are OTHER matches (Bundesliga / 2. Bundesliga 2022/23). Nothing here reads the hackathon
match. Each match has per-frame BallPossession (home/away) and BallStatus (alive/dead) from the
provider, which map straight onto H / A / D.

Source: the pysport mirror of the dataset on Hugging Face, pinned to one commit, and every file is
checked against a SHA-256 so a rebuild always trains on exactly the same bytes. The original
publication is https://doi.org/10.6084/m9.figshare.28196177 if the mirror ever disappears.
"""

import hashlib
import os
import shutil
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "research"))
import dfl_load  # noqa: E402

RAW_DIR = os.path.join(ROOT, "data", "external", "dfl")
NPZ_DIR = os.path.join(ROOT, "data", "external", "dfl_npz")
MIRROR = "https://huggingface.co/datasets/pysport/idsse-data/resolve/{commit}/{name}"
COMMIT = "a715a38dfbaf5f58e431727c2b78d174101a703c"

SHA256 = {
    "DFL_02_01_matchinformation_DFL-COM-000001_DFL-MAT-J03WMX.xml": "a884728647a564cf8b20e5223a1d04a5da6f6725e1e506c8651505327c197ba9",
    "DFL_02_01_matchinformation_DFL-COM-000001_DFL-MAT-J03WN1.xml": "5490286436c86e6f80a538d2b1b379d0efd016e0defcaf594bc59d9ef5079f4c",
    "DFL_02_01_matchinformation_DFL-COM-000002_DFL-MAT-J03WOH.xml": "f23b3ee13b988b1ff9a0bce438c416dac47e71355bbc7ee1dab21aaf3e33c18c",
    "DFL_02_01_matchinformation_DFL-COM-000002_DFL-MAT-J03WOY.xml": "4c497fac683eee8292901c18ca02c0be1d9609d0b9d4bd8aa092c61f9e0b19e9",
    "DFL_02_01_matchinformation_DFL-COM-000002_DFL-MAT-J03WPY.xml": "566dd5e4c86e124ab90fcdba51100881f0436e22afa71832945a451a2d95a420",
    "DFL_02_01_matchinformation_DFL-COM-000002_DFL-MAT-J03WQQ.xml": "e976aa0c5dba862cfa898ea57ec226efce9d1bc04b23b8d775908db86aa6fa41",
    "DFL_02_01_matchinformation_DFL-COM-000002_DFL-MAT-J03WR9.xml": "429b584e89df3e3e8b81f4bd0d004b38bd0b4298bb390b8fb43efccb60963e01",
    "DFL_04_03_positions_raw_observed_DFL-COM-000001_DFL-MAT-J03WMX.xml": "d3dd92e509421a6466acb0a083c009035f1751bc5fd9ef42ffa8f20d01cf7d49",
    "DFL_04_03_positions_raw_observed_DFL-COM-000001_DFL-MAT-J03WN1.xml": "56ef948961a18e150dba0b7b16869d0008ff66f3050a0c355d31fa5b8ec2fcec",
    "DFL_04_03_positions_raw_observed_DFL-COM-000002_DFL-MAT-J03WOH.xml": "ec4dcf30200aad694dea263d5289485fcd0ca42badddaeb7c81f5cd5a4229d4f",
    "DFL_04_03_positions_raw_observed_DFL-COM-000002_DFL-MAT-J03WOY.xml": "a228ef0e676d009cefff1580fcb721b53f598454eb120ec347e0a628a3a4646b",
    "DFL_04_03_positions_raw_observed_DFL-COM-000002_DFL-MAT-J03WPY.xml": "845202c29ad93d696653436da426a7cdaefb848298e07b7343de6529581671e1",
    "DFL_04_03_positions_raw_observed_DFL-COM-000002_DFL-MAT-J03WQQ.xml": "cbefcb0bd7f39f3b94ee69a7bc97aac0f936425062c1cc1b144733df36804d6f",
    "DFL_04_03_positions_raw_observed_DFL-COM-000002_DFL-MAT-J03WR9.xml": "70b63069189019066fb7e3d9db891122edfe787f3a43ba522ebe45ed89ea0269",
}


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def download():
    """Fetch any missing file from the pinned mirror and check every file's SHA-256."""
    os.makedirs(RAW_DIR, exist_ok=True)
    for name, expected in sorted(SHA256.items()):
        path = os.path.join(RAW_DIR, name)
        if not os.path.isfile(path):
            print("downloading {} ...".format(name), flush=True)
            tmp = path + ".part"
            with urllib.request.urlopen(MIRROR.format(commit=COMMIT, name=name)) as src, \
                    open(tmp, "wb") as dst:
                shutil.copyfileobj(src, dst, 1 << 20)
            os.replace(tmp, path)
        if _sha256(path) != expected:
            sys.exit("FAIL  {} does not match its SHA-256; delete it and run again".format(name))
    print("OK       DFL raw files present and verified ({} files)".format(len(SHA256)))


def convert():
    """Raw XML -> one .npz per match (skips matches already converted)."""
    os.makedirs(NPZ_DIR, exist_ok=True)
    for name in sorted(SHA256):
        if "positions_raw_observed" not in name:
            continue
        match = name.rsplit("DFL-MAT-", 1)[1][:-4]
        target = os.path.join(NPZ_DIR, match + ".npz")
        if os.path.isfile(target):
            continue
        info = next(n for n in SHA256 if "matchinformation" in n and n.endswith(match + ".xml"))
        print("converting {} (a few minutes per match) ...".format(match), flush=True)
        data = dfl_load.parse_match(os.path.join(RAW_DIR, info), os.path.join(RAW_DIR, name))
        import numpy as np
        np.savez_compressed(target, **data)
    print("OK       DFL matches converted in {}".format(os.path.relpath(NPZ_DIR, ROOT)))


def ensure():
    """Make sure the converted training matches exist. Called by model.py before training."""
    import bench
    if all(os.path.isfile(os.path.join(NPZ_DIR, m + ".npz")) for m in bench.MATCHES):
        return
    download()
    convert()


if __name__ == "__main__":
    download()
    convert()
