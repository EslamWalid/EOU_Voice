from setuptools import setup, find_packages

setup(
    name="arabic_eou",                   # package name
    version="0.1.0",
    description="Arabic End-of-Utterance detection for LiveKit",
    author="Eslam Walid",
    packages=find_packages(),
    install_requires=[
        "torch",
        "transformers",
        "huggingface-hub",
        "arabert",
        "aiohttp",
        "livekit",
    ],
    python_requires=">=3.10",
)
