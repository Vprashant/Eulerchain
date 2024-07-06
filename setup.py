from setuptools import setup, find_packages

setup(
    name="eulerchain",
    version="0.1.0",
    author="Prashant Verma",
    author_email="prashant27050@gmail.com",
    description="Its a Generative AI Framework for building Q&A and RPA solutions.",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scikit-learn",
        "torch",
        "Pillow",  
        "customtkinter"  
    ],
    entry_points={
        'console_scripts': [
            'eulerdb=gui.app:main' 
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
