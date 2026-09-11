from setuptools import setup, find_packages

setup(
    name="stock_price_predictor_nlp",
    version="0.1.0",
    description="Multimodal Stock Price Predictor combining FinBERT NLP and BiLSTM for Time Series Forecasting",
    author="Anushesh",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "yfinance>=0.2.30",
        "pandas>=2.0.0",
        "numpy>=1.24.0,<2.0.0",
        "scikit-learn>=1.3.0",
        "pyarrow>=12.0.0",
        "tqdm>=4.65.0",
        "pyyaml>=6.0",
    ],
)
