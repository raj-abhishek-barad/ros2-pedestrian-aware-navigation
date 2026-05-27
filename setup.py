from setuptools import find_packages, setup
import os
from glob import glob

package_name = "pedestrian_aware_tb4"

setup(
    name=package_name,
    version="0.2.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        # Required for ament to find the package
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        # Install launch files
        (os.path.join("share", package_name, "launch"), glob("launch/*.py")),
        # Install config files
        (os.path.join("share", package_name, "config"), glob("config/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Your Name",
    maintainer_email="you@example.com",
    description="Pedestrian-aware safety package with IMM tracker and HMM intent estimator.",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "pedestrian_pipeline = pedestrian_aware_tb4.nodes.pedestrian_pipeline:main",
	    "robot_simulator = pedestrian_aware_tb4.nodes.robot_simulator:main",
            "risk_filter         = pedestrian_aware_tb4.nodes.risk_filter:main",
            "fake_scenario       = pedestrian_aware_tb4.nodes.fake_scenario:main",
            "rviz_overlay        = pedestrian_aware_tb4.nodes.rviz_overlay:main",
        ],
    },
)
