import setuptools

if __name__ == "__main__":

    with open('README.md', 'r') as readme_file:
        long_description = readme_file.read()

    with open('requirements.txt', 'r') as f:
        requirements = f.readlines()
        requirements = [line.strip() for line in requirements if line.strip()]

    setuptools.setup(name = 'nanonis_load',
    version = '2.4.2',
    author = 'Dillon Wong',
    author_email = '',
    description = 'Load Nanonis data',
    long_description = long_description,
    long_description_content_type = 'text/markdown',
    url = 'https://github.com/dilwong/nanonis_load',
    install_requires = requirements,
    packages=['nanonis_load'],
    package_dir={'nanonis_load': 'nanonis_load'}
    )