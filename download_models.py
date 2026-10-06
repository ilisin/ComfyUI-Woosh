"""Explicit installer for the three required official T2A checkpoint archives."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import urllib.request
import zipfile

RELEASE = "https://api.github.com/repos/SonyResearch/Woosh/releases/tags/v1.0.0"


def fetch_models(model, destination):
    request = urllib.request.Request(RELEASE, headers={"User-Agent": "ComfyUI-Woosh-Local-installer"})
    with urllib.request.urlopen(request, timeout=60) as response:
        release = json.load(response)
    assets = {asset["name"]: asset for asset in release["assets"]}
    destination.mkdir(parents=True, exist_ok=True)
    for name in (model, "Woosh-AE", "TextConditionerA"):
        target = destination / name
        if all((target / filename).is_file() and (target / filename).stat().st_size > 0 for filename in ("config.yaml", "weights.safetensors")):
            print(f"Already present: {target}", flush=True)
            continue
        asset = assets[f"{name}.zip"]
        archive = destination / f".{name}.download.zip"
        print(f"Downloading {name} ({asset['size'] / 1e9:.2f} GB)...", flush=True)
        digest = hashlib.sha256()
        try:
            with urllib.request.urlopen(asset["browser_download_url"], timeout=120) as response, archive.open("wb") as output:
                while chunk := response.read(8 * 1024 * 1024):
                    output.write(chunk)
                    digest.update(chunk)
            if archive.stat().st_size != asset["size"]:
                raise ValueError(f"Incomplete download: {name}")
            expected_digest = asset.get("digest")
            if expected_digest and expected_digest != f"sha256:{digest.hexdigest()}":
                raise ValueError(f"Checksum mismatch: {name}")
            with zipfile.ZipFile(archive) as source:
                target.mkdir(parents=True, exist_ok=True)
                for filename in ("config.yaml", "weights.safetensors"):
                    matches = [item for item in source.infolist() if not item.is_dir() and Path(item.filename).name == filename]
                    if len(matches) != 1:
                        raise ValueError(f"Expected exactly one {filename} in {name}.zip")
                    # Write only known basenames; do not extract arbitrary archive paths.
                    temporary = target / f"{filename}.part"
                    with source.open(matches[0]) as input_file, temporary.open("wb") as output:
                        shutil.copyfileobj(input_file, output, length=8 * 1024 * 1024)
                    temporary.replace(target / filename)
            print(f"Installed: {target}", flush=True)
        finally:
            archive.unlink(missing_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("Woosh-DFlow", "Woosh-Flow"), default="Woosh-DFlow")
    parser.add_argument("--destination", type=Path, required=True)
    arguments = parser.parse_args()
    fetch_models(arguments.model, arguments.destination.resolve())
