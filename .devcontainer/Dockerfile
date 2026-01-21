# Use a Python 3.12 slim image based on Debian Bookworm
FROM python:3.12-slim-bookworm

# Set environment variables for non-interactive apt-get and Python unbuffered output
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1

# Use a higher-level user or ensure root for install
USER root

# Install system dependencies
# build-essential for compiling some Python packages (e.g., numpy, scipy)
# libatlas-base-dev for optimized linear algebra (BLAS/LAPACK) for statsmodels
# Playwright dependencies: libicu-dev, libwoff2-dev, libatk-bridge2.0-0, libxkbcommon-x11-0, libgtk-3-0
# curl and ca-certificates for downloading rustup
# Clean and update with better error handling
RUN apt-get clean && \
    apt-get update --fix-missing && \
    apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    ca-certificates \
    git \
    libatlas-base-dev \
    libicu-dev \
    # These are often the culprits in slim images:
    libatk-bridge2.0-0 \
    libxkbcommon-x11-0 \
    libgtk-3-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Rust for Polars optimization
# This installs rustup and the default toolchain
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Set the working directory
WORKDIR /app

# Copy requirements file and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install Playwright browsers (chromium, firefox, webkit)
RUN playwright install --with-deps

# Copy the rest of the application code
COPY . .

# Expose any ports if necessary (e.g., for Jupyter, Streamlit)
# EXPOSE 8888

# Command to run the application (can be overridden by devcontainer.json or docker-compose)
CMD ["python", "main.py"]
