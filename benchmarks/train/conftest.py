def pytest_configure(config):
    config.addinivalue_line("markers", "slow: CPU smoke test that downloads a model and trains it")
