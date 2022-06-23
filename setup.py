from setuptools import setup, find_packages
from pathlib import Path

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="gale-topo",
    version="0.0.1",
    packages=find_packages(),
    # Project uses reStructuredText, so ensure that the docutils get
    # installed or upgraded on the target machine
    install_requires=[
        "gudhi>=3.4.1.post1",
        "kmapper>=2.0.1",
        "numpy>=1.19.5",
        "networkx>=2.6.3",
        "scikit-learn>=0.24.2",
    ],
    # metadata to display on PyPI
    author="Peter Xenopoulos, Gromit Chan, ",
    author_email="xenopoulos@nyu.edu",
    description="Globally Assessing Local Explanations (GALE), a method to compare local explaination output",
    keywords="explainability interpretability topology",
    url="https://github.com/pnxenopoulos/gale",
    project_urls={
        "Issues": "https://github.com/pnxenopoulos/gale/issues",
        "Documentation": "https://github.com/pnxenopoulos/gale/tree/main/docs",
        "GitHub": "https://github.com/pnxenopoulos/gale/",
    },
    classifiers=["License :: OSI Approved :: MIT License"],
)
