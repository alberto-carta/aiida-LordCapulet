from setuptools import setup, find_packages

setup(
    name='lordcapulet',
    version='0.1.0',
    packages=find_packages(),
    entry_points={
        'aiida.calculations': [
            'constrained_pw = lordcapulet.constrained_pw:CustomPwCalculation',
        ],
        'aiida.workflows': [
            'lordcapulet.afm_scan = lordcapulet_workflows.simple_afm_scan:AFMScanWorkChain',
        ],
    },
    install_requires=['aiida-core', 'aiida-quantumespresso'],
)
