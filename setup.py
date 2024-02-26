from setuptools import setup, find_packages

setup(
    name="VeinSegmentation",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "contourpy==1.2.0",
        "cycler==0.12.1",
        "fonttools==4.49.0",
        "imageio==2.34.0",
        "kiwisolver==1.4.5",
        "lazy_loader==0.3",
        "matplotlib==3.8.3",
        "networkx==3.2.1",
        "numpy==1.26.4",
        "packaging==23.2",
        "pillow==10.2.0",
        "pyparsing==3.1.1",
        "python-dateutil==2.8.2",
        "scikit-image==0.22.0",
        "scipy==1.12.0",
        "six==1.16.0",
        "tifffile==2024.2.12"
    ]
)