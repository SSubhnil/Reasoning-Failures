def pytest_configure(config):
    config.addinivalue_line(
        "markers", "network: test requires network/HF Hub access (skip offline)"
    )
