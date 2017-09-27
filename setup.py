from setuptools import setup

setup(
    install_requires=['docopt>=0.6.1', 'coverage>=3.6', 'requests>=1.0.0'],
    setup_requires=['pytest-runner'],
    tests_require=[ 'pytest']#'mock','sh>=1.08'
)

