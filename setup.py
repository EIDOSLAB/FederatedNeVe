from setuptools import setup, find_packages

setup(
    name="my_federated",  # Nome del pacchetto
    version="1.0",
    packages=find_packages(where="src"),  # Cerca pacchetti nella directory "src"
    package_dir={"": "src"},  # Mappa "src" come directory di riferimento
    install_requires=[
        # Elenca le dipendenze qui
    ],
    include_package_data=True,  # Include file aggiuntivi specificati
    classifiers=[
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
