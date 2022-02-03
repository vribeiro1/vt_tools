from setuptools import setup, find_packages

requirements = [
    "funcy",
    "matplotlib",
    "numba",
    "numpy==1.21.*",
    "scipy",
    "torch",
    "ujson"
]

setup(
    name="vt_tools",
    version="0.0.1",
    description="Vocal tract tools",
    author="Vinicius Ribeiro",
    author_email="vinicius.souza-ribeiro@loria.fr",
    license="MIT",
    packages=find_packages(),
    zip_safe=False,
    install_requirements=requirements
)
