# Official NVIDIA CUDA base image con Ubuntu 22.04
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Disable interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# 1. Python and Java installation
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3.10-venv \
    openjdk-17-jdk-headless \
    curl \
    git \
    procps \
    && rm -rf /var/lib/apt/lists/*

# 2. Configure Java and Spark
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV SPARK_VERSION=3.5.0
ENV HADOOP_VERSION=3
ENV SPARK_HOME=/opt/spark
ENV PATH=$PATH:$SPARK_HOME/bin:$SPARK_HOME/sbin

# Download and install Spark
RUN curl -o spark.tgz https://archive.apache.org/dist/spark/spark-${SPARK_VERSION}/spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz \
    && tar -xf spark.tgz -C /opt/ \
    && mv /opt/spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION} /opt/spark \
    && rm spark.tgz

# 3. Configure Python
RUN ln -s /usr/bin/python3.10 /usr/bin/python
ENV PYSPARK_PYTHON=/usr/bin/python3
ENV PYSPARK_DRIVER_PYTHON=/usr/bin/python3

# 4. Install PyTorch (GPU Version to avoid errors)
RUN pip3 install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 5. Install all other project libraries
RUN pip3 install --no-cache-dir \
    pyspark==3.5.0 \
    transformers \
    accelerate \
    datasets \
    peft \
    evaluate \
    scikit-learn \
    pandas \
    pyarrow \
    numpy \
    psutil \
    pyyaml \
    pillow \
    tqdm

# Working directory
WORKDIR /app

# Default command: keep the container running
CMD ["tail", "-f", "/dev/null"]