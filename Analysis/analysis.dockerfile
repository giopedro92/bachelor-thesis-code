# Use root docker as base image (on Ububntu with root and python)
FROM docker.io/rootproject/root

# Update the system packages and upgrade existing packages
RUN apt update && apt install -y --no-install-recommends \
    python3-pip python3-venv\
    && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
RUN pip install numpy matplotlib uproot awkward