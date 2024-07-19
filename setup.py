import setuptools

# setuptools.setup(packages=['my_chess'])
setuptools.setup(
    # ...
    packages=setuptools.find_packages(
        where='.',
        include=['my_chess'],  # alternatively: `exclude=['additional*']`
    ),
    package_dir={"": "."}
    # ...
)