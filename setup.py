from setuptools import setup, find_packages
NAME = 'evapotranspiration'

VERSION = '0.0.1'

DESCRIPTION = 'evapotranspiration'

LONG_DESCRIPTION = 'Reference evapotranspiration'

CLASSIFIERS = [  # https://pypi.python.org/pypi?:action=list_classifiers
    'Development Status :: 4 - Beta',
    'Environment :: Console',
    'Intended Audience :: Science/Research',
    'License :: OSI Approved :: MIT License',
    'Natural Language :: English',
    'Operating System :: OS Independent',
    'Programming Language :: Python :: 3 :: Only',
    'Topic :: Scientific/Engineering :: Hydrology'
]

setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    download_url='https://github.com/JRoehrig/evapotranspiration',
    url='https://github.com/JRoehrig/evapotranspiration',
    author='Jackson Roehrig',
    author_email='Jackson.Roehrig@th-koeln.de',
    license='MIT',
    classifiers=CLASSIFIERS,
    install_requires=['pandas'],
    packages=find_packages(),
    scripts=[]
)

