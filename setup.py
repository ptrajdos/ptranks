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
                'numpy>=1.22.4',
                'scipy>=1.10.1',
        ],
        test_suite='test'
        )
