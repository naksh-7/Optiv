# SecureDoc AI Setup Script
from setuptools import setup, find_packages
from pathlib import Path

# Read README file
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

# Read requirements
requirements_path = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_path.exists():
    with open(requirements_path, "r", encoding="utf-8") as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="securedoc-ai",
    version="1.0.0",
    author="SecureDoc AI Team",
    author_email="team@securedoc-ai.com",
    description="Advanced Document PII Detection & Logo Removal System",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/securedoc-ai",
    
    # Package configuration
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    
    # Dependencies
    install_requires=requirements,
    
    # Optional dependencies
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "pytest-mock>=3.11.0",
            "black>=23.7.0",
            "flake8>=6.0.0",
            "mypy>=1.5.0",
            "pre-commit>=3.3.0",
        ],
        "docs": [
            "sphinx>=7.1.0",
            "sphinx-rtd-theme>=1.3.0",
            "myst-parser>=2.0.0",
        ],
        "streamlit": [
            "streamlit>=1.25.0",
            "plotly>=5.15.0",
            "altair>=5.0.0",
        ],
        "api": [
            "flask>=2.3.0",
            "flask-restx>=1.1.0",
            "flask-cors>=4.0.0",
            "gunicorn>=21.2.0",
        ]
    },
    
    # Entry points for command line tools
    entry_points={
        "console_scripts": [
            "securedoc=src.cli:main",
            "securedoc-api=src.api.app:main",
            "securedoc-streamlit=src.ui.streamlit_app:main",
            "securedoc-train=src.models.train:main",
        ],
    },
    
    # Package data
    package_data={
        "": [
            "config/*.yaml",
            "config/*.yml",
            "config/*.json",
            "models/*.pt",
            "models/*.pth",
            "assets/*",
        ],
    },
    include_package_data=True,
    
    # Metadata
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Information Technology",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Security",
        "Topic :: Text Processing :: Linguistic",
        "Topic :: Multimedia :: Graphics :: Graphics Conversion",
    ],
    
    keywords=[
        "pii-detection", "privacy", "security", "ocr", "computer-vision",
        "document-processing", "anonymization", "yolo", "nlp", "ai"
    ],
    
    project_urls={
        "Homepage": "https://github.com/yourusername/securedoc-ai",
        "Documentation": "https://securedoc-ai.readthedocs.io/",
        "Repository": "https://github.com/yourusername/securedoc-ai",
        "Issues": "https://github.com/yourusername/securedoc-ai/issues",
        "Changelog": "https://github.com/yourusername/securedoc-ai/releases",
    },
    
    # Zip safe
    zip_safe=False,
)