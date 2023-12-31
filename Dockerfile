# Start from an NVIDIA CUDA base image
FROM nvidia/cuda:12.3.1-base-ubuntu20.04

# Install asdf dependencies
RUN apt-get update -y && apt-get install -y \
  curl \
  git \
  build-essential

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=America/Los_Angeles

RUN apt-get update && apt-get install -y python3.10

RUN apt-get install -y python3-pip

# Install KServe and its dependencies
RUN pip install kserve transformers torch \
  --extra-index-url https://download.pytorch.org/whl/cu123

RUN pip install accelerate

# Set the working directory in the container
WORKDIR /app

# Copy the Python script into the container
COPY . /app

# Expose the port the server will run on
EXPOSE 8080

# Set the command to run the model server
CMD ["python3", "kserveme.py"]
