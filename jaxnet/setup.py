from setuptools import setup, find_packages

setup(
    name="jaxnet",
    version="0.0.1dev0",
    url="https://github.com/glemaitre/jaxnet.git",
    author="Guillaume Lemaitre",
    author_email="g.lemaitre58@gmail.com",
    description="A tot neural network library to learn JAX",
    packages=find_packages(),
    install_requires=[
        "dm-haiku",
        "jax",
        "jaxlib",
        "optax",
    ],
    extra_requies={
        "tests": [
            "pytest",
        ],
    },
)
