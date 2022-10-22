import setuptools

if __name__ == "__main__":

    with open('requirements.txt', 'r') as f:
        requirements = f.readlines()
        requirements = [line.strip() for line in requirements if line.strip()]

    setuptools.setup(name = 'nanonis_load',
    version = '2.4.0',
    author = 'Dillon Wong',
    author_email = '',
    description = 'Load Nanonis data',
    url = 'https://github.com/dilwong/nanonis_load',
    install_requires = requirements,
    packages=['nanonis_load'],
    package_dir={'nanonis_load': 'nanonis_load'}
    )