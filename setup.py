import re
from pathlib import Path
from setuptools import setup, find_packages

def get_version():
    """
    Lê a versão diretamente do pyproject.toml para evitar duplicação.
    """
    content = Path("pyproject.toml").read_text()
    match = re.search(r'^version\s*=\s*["\'](.+)["\']', content, re.MULTILINE)
    if match:
        return match.group(1)
    raise RuntimeError("Não foi possível encontrar a versão no pyproject.toml")

setup(
    name="modeltrack",
    version=get_version(),
    description="Statistical analysis of experimental designs in Python",
    author="Cristiano F. Oliveira",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "pandas",
        "numpy",
        "scipy",
        "statsmodels",
        "scikit-posthocs",
        "matplotlib",
        "seaborn",
        "tabulate",
        "pyspark"
    ],
)