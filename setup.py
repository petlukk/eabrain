from setuptools import setup

setup(
    name="eabrain",
    version="0.1.0",
    py_modules=["eabrain", "indexer", "memory", "text_search"],
    install_requires=["numpy>=1.21"],
    extras_require={"dev": ["pytest>=7.0"]},
    entry_points={"console_scripts": ["eabrain=eabrain:main"]},
    author="Peter Lukka",
    author_email="peter.lukka@gmail.com",
    description="Eä-driven context engine for Claude Code",
)
