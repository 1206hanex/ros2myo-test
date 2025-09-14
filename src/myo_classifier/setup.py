from setuptools import setup, find_packages
from glob import glob
import os

package_name = 'myo_classifier'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
    ],
    install_requires=[
        'setuptools',
        'bleak',
        'numpy',
        'tensorflow',         # only if you will use the CNN subscriber
        'scikit-learn'        # only if you will use RF/SVM/KNN
    ],
    zip_safe=True,
    maintainer='hanex',
    maintainer_email='hans.lomboy1206@gmail.com',
    description='Myo publisher and gesture classifiers (RF/CNN)',
    license='MIT',
    entry_points={
        'console_scripts': [
            'myo_stream = myo_classifier.myo_stream:main',
            'myo_rf     = myo_classifier.myo_rf:main',
            'myo_cnn    = myo_classifier.myo_cnn:main',
            'manager = myo_classifier.manager:main',
        ],
    },
)
