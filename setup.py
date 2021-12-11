from setuptools import setup

setup(
    name = "satlasso",
    version = "0.1.0",
    author = "Vanessa D. Jonsson, Natalie Dullerud",
    author_email = "vjonsson@ucsc.edu, dullerud@usc.edu",
    packages = ["satlasso", "satlasso.test"],
    license = "LICENSE.txt",
    description = "A package for LASSO regression on partially saturated data",
    long_description = open("README.md").read(),
    install_requires = [
        "numpy >= 1.18.0",
        "scikit-learn >= 0.23.0",
        "cvxpy",
        "statistics"
    ]
    )
