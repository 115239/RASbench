# Runner

The runner contains `robolab_runner/` (CLI and recording protocol), `m20_runtime/` (simulation, safety monitoring, and intervention integration), and vendored protocol adapters.

Runtime dependencies are declared in [pyproject.toml](pyproject.toml). Full execution requires the matching Isaac Sim environment, robot and scene assets, executable task contracts, and a policy endpoint.

The source test suite is in [tests](tests); simulator-dependent tests require the original runtime dependencies.
