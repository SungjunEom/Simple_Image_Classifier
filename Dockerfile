FROM nvidia/cuda:12.2.0-runtime-ubuntu20.04

MAINTAINER Jun <junjun607@uos.ac.kr>

RUN \
	apt-get update && \
	apt-get install -y wget git python3-venv \
	libgoogle-perftools4 libtcmalloc-minimal4 python3-pip

RUN	apt-get install -y python-is-python3

RUN	pip install notebook matplotlib scikit-image

RUN	jupyter notebook --generate-config && \
	echo "c.NotebookApp.allow_origin = '*'" >> /root/.jupyter/jupyter_notebook_config.py && \
	echo "c.NotebookApp.ip = '0.0.0.0'" >> /root/.jupyter/jupyter_notebook_config.py

COPY	./requirements.txt /tmp

RUN	pip install -r /tmp/requirements.txt

RUN 	pip install wandb

WORKDIR /workspace

CMD ["bash"]
