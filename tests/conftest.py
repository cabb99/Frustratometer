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
    parser.addoption(
        "--skip-stochastic",
        action="store_true",
        default=False,
        help="Skip tests marked as stochastic (Monte Carlo / random-sampling results).",
    )
    parser.addoption(
        "--skip-memory-heavy",
        action="store_true",
        default=False,
        help="Skip tests marked as memory_heavy (peak RSS > 7 GB).",
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
    config.addinivalue_line(
        "markers",
        "stochastic: marks tests whose results depend on random sampling and may "
        "occasionally fail; runs by default, pass --skip-stochastic to exclude.",
    )
    config.addinivalue_line(
        "markers",
        "memory_heavy: marks tests that require more than ~7 GB of RAM "
        "(e.g. large protein AWSEM models); "
        "runs by default, pass --skip-memory-heavy to exclude.",
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

    if config.getoption("--skip-stochastic"):
        skip_stochastic = pytest.mark.skip(reason="stochastic test — skipped via --skip-stochastic")
        for item in items:
            if "stochastic" in item.keywords:
                item.add_marker(skip_stochastic)

    if config.getoption("--skip-memory-heavy"):
        skip_mh = pytest.mark.skip(reason="memory-heavy test — skipped via --skip-memory-heavy")
        for item in items:
            if "memory_heavy" in item.keywords:
                item.add_marker(skip_mh)
