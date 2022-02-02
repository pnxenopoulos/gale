from setuptools import setup, find_packages

setup(
    name="gale",
    version="1.0",
    packages=find_packages(),
    # Project uses reStructuredText, so ensure that the docutils get
    # installed or upgraded on the target machine
    install_requires=[
        "numpy>=1.18.1",
    ],
    # metadata to display on PyPI
    author="Peter Xenopoulos, Gromit Chan, ",
    author_email="xenopoulos@nyu.edu",
    description="Globally Assessing Local Explanations (GALE)",
    keywords="explainability interpretability",
    url="https://github.com/pnxenopoulos/gale",
    project_urls={
        "Issues": "https://github.com/pnxenopoulos/gale/issues",
        "Documentation": "https://github.com/pnxenopoulos/gale/tree/main/docs",
        "GitHub": "https://github.com/pnxenopoulos/gale/",
    },
    classifiers=["License :: OSI Approved :: MIT License"],
)
