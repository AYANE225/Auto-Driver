from setuptools import find_packages, setup

package_name = "av_perception"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="cst",
    maintainer_email="dev@example.com",
    description="ROS 2 nodes wrapping the perception_core detect-track-predict pipeline.",
    license="MIT",
    entry_points={
        "console_scripts": [
            "perception_node = av_perception.perception_node:main",
            "synthetic_publisher = av_perception.synthetic_publisher_node:main",
        ],
    },
)
