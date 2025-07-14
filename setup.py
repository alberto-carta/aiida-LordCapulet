from setuptools import setup, find_packages

setup(
    name='lordcapulet',
    version='0.1.0',
    packages=find_packages(),
    entry_points={
        'aiida.calculations': [
            'constrained_pw = lordcapulet_custom_pw.custom_pw:ConstrainedPWCalculation',
        ],
        'aiida.workflows': [
            'lordcapulet.afm_scan = lordcapulet_workflows.simple_afm_scan:AFMScanWorkChain',
            'lordcapulet.constrained_scan = lordcapulet_workflows.constrained_scan:ConstrainedScanWorkChain',
        ],
    },
    install_requires=['aiida-core', 'aiida-quantumespresso'],
)
