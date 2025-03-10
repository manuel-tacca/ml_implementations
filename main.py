import unittest

if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    tests = [
        "logistic_regression.test.test_logistic_regression",
        "linear_regression.test.test_linear_regression",
        "neural_network.test.test_neural_network",
        "neural_network.test.test_neural_network_multi"
    ]
    
    for test in tests:
        suite.addTests(loader.loadTestsFromName(test))

    runner = unittest.TextTestRunner()
    runner.run(suite)
