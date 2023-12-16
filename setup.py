import sys
import setuptools

# READ README.md for long description on PyPi.
try:
    long_description = open("README.md", encoding="utf-8").read()
except Exception as e:
    sys.stderr.write(f"Failed to read README.md:\n  {e}\n")
    sys.stderr.flush()
    long_description = ""

setuptools.setup(
    name="independent_component_analysis",
    author="Daniel Bonnery, Augustin Poissonnier, Paul Guillermit, Yvann Le Fay",
    description="Accelerating Bayesian Estimation for Network Poisson Models Using Frequentist Variational Estimates ",
    long_description=long_description,
    version="0.1",
    packages=setuptools.find_packages(),
    install_requires=[
        "numpy",
        "scipy",
        "urllib"
        "pandas"
    ],
    long_description_content_type="text/markdown",
    keywords="SMC poisson network regression VAE bayesian variational inference ",
    license="MIT",
    license_files=("LICENSE",),
)
