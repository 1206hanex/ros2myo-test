from setuptools import find_packages, setup
import glob
import os

package_name = 'myo_space_trigger'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            [os.path.join('resource', package_name)]),
        (f'share/{package_name}', ['package.xml']),
        # install your launch files under share/myo_space_trigger/launch
        (f'share/{package_name}/launch', glob.glob('launch/*.py')),
    ],
    package_data={
        'myo_space_trigger': ['launch/*.py'],
    },
    install_requires=[
        'setuptools',
        'bleak',
        'pyautogui',
        'numpy',
    ],
    zip_safe=True,
    maintainer='hanex',
    maintainer_email='hans.lomboy1206@gmail.com',
    description='ROS2 nodes for Myo armband EMG & gesture classification',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'myo_listener = myo_space_trigger.myo_listener:main',
            'myo_subscriber = myo_space_trigger.myo_subscriber:main',
            'myo_stream = myo_space_trigger.myo_stream:main',
            'myo_rf = myo_space_trigger.myo_rf:main',
            'myo_cnn = myo_space_trigger.myo_cnn:main',
        ],
    },
)
