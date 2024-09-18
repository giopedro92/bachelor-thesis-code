FROM docker.io/rootproject/root

RUN apt update && apt install -y --no-install-recommends \
    python3-pip python3-venv\
    && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install numpy matplotlib uproot awkward