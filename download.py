"""Download versioned RASBench release assets and verify SHA-256 (no login)."""
from pathlib import Path
import argparse
import hashlib
import json
import urllib.request

ROOT = Path(__file__).resolve().parent
REPOSITORY_ID = 1388304250
RELEASE_ID = 442546855


def request(url, accept):
    return urllib.request.urlopen(urllib.request.Request(
        url, headers={'Accept': accept, 'User-Agent': 'RASBench-materials'}), timeout=120)


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('group', choices=['scenes', 'adapters'])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.group == 'scenes':
        expected = json.loads((ROOT / 'scenes/downloads/SCENE_ARCHIVES.json').read_text())['parts']
    else:
        line = (ROOT / 'our_agent/experiments/market_offline/ADAPTER_SHA256SUMS.txt').read_text().strip()
        digest, name = line.split(maxsplit=1)
        expected = [{'name': name, 'sha256': digest}]
    api = f'https://api.github.com/repositories/{REPOSITORY_ID}'
    with request(f'{api}/releases/{RELEASE_ID}', 'application/vnd.github+json') as response:
        assets = {item['name']: item for item in json.load(response)['assets']}
    args.output.mkdir(parents=True, exist_ok=True)
    for item in expected:
        name = item['name']
        if Path(name).name != name:
            raise ValueError('Invalid asset filename')
        destination = args.output / name
        if destination.exists():
            if sha256(destination) != item['sha256']:
                raise ValueError(f'Existing file has a different checksum: {destination}')
            print(f'Verified {name}', flush=True)
            continue
        asset = assets[name]
        partial = destination.with_suffix(destination.suffix + '.part')
        print(f'Downloading {name} ({asset["size"]} bytes)', flush=True)
        with request(f'{api}/releases/assets/{asset["id"]}', 'application/octet-stream') as response:
            if response.headers.get('Content-Type', '').startswith('application/json'):
                raise RuntimeError('GitHub returned metadata instead of file content')
            with partial.open('wb') as stream:
                while block := response.read(8 * 1024 * 1024):
                    stream.write(block)
        if partial.stat().st_size != asset['size'] or sha256(partial) != item['sha256']:
            raise ValueError(f'Integrity check failed: {partial}')
        partial.replace(destination)
        print(f'Verified {name}', flush=True)


if __name__ == '__main__':
    main()
