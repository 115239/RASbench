"""Build the self-contained RoboLab V2 source distribution.

The historical repository stores the ``v2`` package at this project root.
Setuptools therefore maps this directory to the installed package name
``v2`` and installs the frozen runtime dependencies from ``vendor`` as their
original top-level package names.
"""

from __future__ import annotations

from pathlib import Path

from setuptools import find_packages, setup


PROJECT_ROOT = Path(__file__).resolve().parent


def _v2_packages() -> list[str]:
    children = find_packages(
        where=str(PROJECT_ROOT),
        exclude=(
            "vendor",
            "vendor.*",
            "tests",
            "tests.*",
        ),
    )
    return ["v2", *("v2." + name for name in children)]


def _vendor_packages() -> list[str]:
    return find_packages(where=str(PROJECT_ROOT / "vendor"))


setup(
    packages=[*_v2_packages(), *_vendor_packages()],
    package_dir={
        "v2": ".",
        **{
            package: "vendor/" + package.replace(".", "/")
            for package in _vendor_packages()
        },
    },
    include_package_data=True,
    package_data={
        "*": ["*.json", "resources/**/*.json", "resources/**/*.md"],
        "v2.common.rule_package_v1": ["schema.json"],
    },
)
