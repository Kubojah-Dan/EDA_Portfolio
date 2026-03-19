from setuptools import find_packages, setup

setup(
    name="ml_eda_portfolio",
    version="1.0.0",
    description="US Accidents EDA & ML Portfolio",
    author="Your Name",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
    install_requires=[
        "pandas>=2.2",
        "numpy>=1.26",
        "scikit-learn>=1.4",
        "xgboost>=2.0",
        "lightgbm>=4.3",
        "fastapi>=0.111",
        "dash>=2.17",
        "sqlalchemy>=2.0",
    ],
)

