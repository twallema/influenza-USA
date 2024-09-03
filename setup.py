from setuptools import find_packages, setup

setup(
    name='influenza_USA',
    packages=find_packages("src", exclude=["*.tests"]),
    package_dir={'': 'src'},
    version='0.0',
    description='An age- and spatially stratified influenza model for the USA',
    author='Tijs Alleman, Johns Hopkins University',
    license='CC-BY-NC-SA',
    install_requires=[
        'pySODM',
        'tensorflow'
    ],
)
