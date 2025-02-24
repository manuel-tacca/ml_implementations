import unittest

if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    tests = [
        "logistic_regression.test.test_logistic_regression",
        "linear_regression.test.test_linear_regression",
    ]
    
    for test in tests:
        suite.addTests(loader.loadTestsFromName(test))

    runner = unittest.TextTestRunner()
    runner.run(suite)
