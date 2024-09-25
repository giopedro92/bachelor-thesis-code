# Use Ubuntu 22.04 as the base image
FROM ubuntu:22.04

# Set the working directory inside the container
WORKDIR /app

# Use bash as the default shell
SHELL ["/bin/bash", "-c"]

# Update the system packages and upgrade existing packages
RUN apt-get update && apt-get upgrade -y

# Install graphviz for graph visualization
RUN apt-get install -y graphviz

# Install the Vim text editor
RUN apt-get install -y vim

# Install Python 3 and pip
RUN apt-get install -y python3 python3-pip

# Install pkg-config and HDF5 dependencies
RUN apt-get install -y pkg-config libhdf5-dev

# Upgrade pip and install Python dependencies
RUN pip3 install --upgrade pip
RUN pip3 install scikit-learn matplotlib pandas numpy seaborn
RUN pip3 install uproot tensorflow

# Specify to start a bash shell when the container starts
CMD ["/bin/bash"]