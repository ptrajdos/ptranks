from setuptools import setup, find_packages



setup(
        name='ptranks',
        version ='0.0.1',
        author='Pawel Trajdos',
        author_email='pawel.trajdos@pwr.edu.pl',
        url = 'https://github.com/ptrajdos/ptranks',
        description="Procedures for calculate rankings",
        packages=find_packages(include=[
                'ptranks',
                'ptranks.*',
                ]),
        install_requires=[ 
                'scipy>=1.10.1',
        ],
        extras_require={
        "dev": [
                'coverage>=7.8.0, <8.0.0',
                'unittest-parallel>=1.5.3, <2.0.0',
                'pandas>=2.0.0',
                'pdoc3>=0.11.1, <1.0.0',
                'pytest>=8.3.5, <9.0.0',
                'pytest-cov>=6.1.1, <7.0.0',
                'pytest-profiling>=1.8.1, <2.0.0',
                'pytest-xdist>=3.6.1, <4.0.0',
                'snakeviz>=2.2.2, <3.0.0',
                'tox>=4.0.0, <5.0.0',
        ],
        },
        test_suite='test'
        )
