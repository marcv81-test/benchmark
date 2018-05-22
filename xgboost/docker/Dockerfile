FROM ubuntu:18.04

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update && apt-get install -y \
git \
build-essential \
python3 \
python3-setuptools

RUN cd /root \
&& git clone --recursive --branch v0.71 https://github.com/dmlc/xgboost.git \
&& cd xgboost \
&& ./build.sh \
&& cd python-package \
&& python3 setup.py install

COPY benchmark.py /root

WORKDIR /root
CMD /bin/bash
