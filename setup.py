from setuptools import setup, find_packages

setup(
    name='lordcapulet',
    version='0.1.0',
    packages=find_packages(),
    entry_points={
        'aiida.calculations': [
            'lordcapulet.constrained_pw = lordcapulet.calculations.constrained_pw:ConstrainedPWCalculation',
        ],
        'aiida.workflows': [
            'lordcapulet.afm_scan = lordcapulet.workflows.afm_scan:AFMScanWorkChain',
            'lordcapulet.constrained_scan = lordcapulet.workflows.constrained_scan:ConstrainedScanWorkChain',
        ],
    },
    install_requires=[
        'aiida-core>=2.0.0',
        'aiida-quantumespresso>=4.0.0',
        'numpy',
    ],
    python_requires='>=3.8',
    author='Alberto Carta',
    author_email='your.email@example.com',
    description='AiiDA plugins for constrained DFT+U calculations',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/your-username/aiida-lordcapulet',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Framework :: AiiDA',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: POSIX :: Linux',
        'Operating System :: MacOS :: MacOS X',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Scientific/Engineering :: Chemistry',
        'Topic :: Scientific/Engineering :: Physics',
    ],
)
