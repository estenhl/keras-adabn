import os
import sys


def pytest_configure(config):
    testpath = os.path.dirname(os.path.abspath(__file__))
    libpath = os.path.join(testpath, os.pardir)
    sys.path.append(libpath)
