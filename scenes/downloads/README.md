# Scene media downloads

The six independent ZIP archives contain 1,124 screenshots and eight tour videos, hosted in GitHub Releases. The downloader uses the asset API and verifies each archive's SHA-256.

From the repository root:

```bash
python download.py scenes --output scene_archives
mkdir -p scene_media
for archive in scene_archives/*.zip; do unzip -o "$archive" -d scene_media; done
```

| Archive | Bytes |
| --- | ---: |
| `RASBench-scenes-01.zip` | 1795356604 |
| `RASBench-scenes-02.zip` | 1796355602 |
| `RASBench-scenes-03.zip` | 1575197801 |
| `RASBench-scenes-04.zip` | 1794590404 |
| `RASBench-scenes-05.zip` | 1799927365 |
| `RASBench-scenes-06.zip` | 1133897666 |

[File list](SCENE_FILES.csv) · [Checksums](SCENE_SHA256SUMS.txt)
