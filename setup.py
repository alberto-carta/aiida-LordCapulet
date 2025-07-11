from setuptools import setup, find_packages

setup(
    name='lordcapulet',
    version='0.1.0',
    packages=find_packages(),
    entry_points={
        'aiida.calculations': [
            'constrained_pw = aiida_custom_pw.custom_pw:CustomPwCalculation',
        ],
    },
    install_requires=['aiida-core', 'aiida-quantumespresso'],
)
