"""
Setup configuration for science-agent-sdk.
"""
from setuptools import setup, find_packages

# 读取README文件
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="science-agent-sdk",
    version="0.1.2",
    description="SDK for science agent and mcp tools",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Digital Pathology Team",
    # author_email="team@digitalpathology.com",
    url="https://github.com/dptech-corp/science-agent-sdk/",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    include_package_data=True,
    python_requires=">=3.10",
    install_requires=[
        "click",
        "mcp", 
        "paho-mqtt>=1.6.1",      # MQTT客户端
        "redis>=5.0.1",          # Redis客户端，使用最新稳定版
        "python-dotenv>=1.0.0",  # 环境变量管理
        "typing-extensions>=4.8.0",  # 额外的类型提示功能
        "aiohttp>=3.9.1",        # 异步HTTP客户端
        "dpdispatcher",         #DPDispatcher
        "lbg",             #lbg
    ],
    # extras_require={
    #     "dev": [
    #         "pytest>=7.4.0",
    #         "pytest-asyncio>=0.23.0",
    #         "pytest-cov>=4.1.0",
    #         "black>=23.11.0",
    #         "isort>=5.12.0",
    #         "mypy>=1.7.0",
    #         "pylint>=3.0.0",
    #     ],
    #     "docs": [
    #         "sphinx>=7.2.0",
    #         "sphinx-rtd-theme>=1.3.0",
    #     ],
    # },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",  # 明确表示仅支持Python 3
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Typing :: Typed",
    ],
    entry_points={
        "console_scripts": [
            "dp-agent=dp.agent.cli.cli:main",
            "dp-agent-lab=dp.agent.lab.main:main",
            "dp-agent-cloud=dp.agent.cloud.main:main",
        ],
    },
    project_urls={
        "Bug Reports": "https://github.com/dptech-corp/science-agent-sdk/issues",
        "Source": "https://github.com/dptech-corp/science-agent-sdk/",
        # "Documentation": "https://science-agent-sdk.readthedocs.io/",
    },
)
