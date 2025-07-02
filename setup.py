from setuptools import setup, find_packages

setup(
    name="side-by-side-comparison",
    version="1.0.0",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "streamlit>=1.28.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "ydata-profiling>=4.6.0",
        "openpyxl>=3.1.2",
        "xlsxwriter>=3.1.9",
        "chardet>=5.2.0",
        "requests>=2.31.0",
        "sqlalchemy>=2.0.0",
        "teradatasql>=17.20.0.0",
        "python-dotenv>=1.0.0"
    ],
    entry_points={
        'console_scripts': [
            'comparison-tool=app:main',
        ],
    },
)
