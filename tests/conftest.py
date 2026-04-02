import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--skip-slow",
        action="store_true",
        default=False,
        help="Skip tests marked as slow (brute-force / expensive computations).",
    )
    parser.addoption(
        "--skip-network",
        action="store_true",
        default=False,
        help="Skip tests that require network access (downloads, remote databases).",
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "slow: marks tests as slow (brute-force or expensive computations); "
        "runs by default, pass --skip-slow to exclude.",
    )
    config.addinivalue_line(
        "markers",
        "network: marks tests that require network access; "
        "runs by default, pass --skip-network to exclude.",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--skip-slow"):
        skip_slow = pytest.mark.skip(reason="slow test — skipped via --skip-slow")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)

    if config.getoption("--skip-network"):
        skip_network = pytest.mark.skip(reason="network test — skipped via --skip-network")
        for item in items:
            if "network" in item.keywords:
                item.add_marker(skip_network)
