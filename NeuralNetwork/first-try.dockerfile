# Use root docker as base image (on Ububntu with ROOT and python)
FROM docker.io/rootproject/root

# Update the system packages and upgrade existing packages
RUN apt-get update && apt-get upgrade -y

# Update the system packages and upgrade existing packages
RUN apt update && apt install -y --no-install-recommends \
    python3-pip python3-venv\
    && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install graphviz for graph visualization
RUN apt-get install -y graphviz

# Install pkg-config and HDF5 dependencies
RUN apt-get install -y pkg-config libhdf5-dev

# Install Python dependencies
RUN pip install numpy matplotlib uproot awkward
RUN pip install scikit-learn tensorflow pandas seaborn