FROM ubuntu:18.04

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update && apt-get install -y \
git \
build-essential \
kernel-package \
fakeroot \
libncurses5-dev \
libelf-dev \
libssl-dev

RUN cd /root && curl https://codeload.github.com/torvalds/linux/tar.gz/v4.15 | tar xvz

WORKDIR /root/linux-4.15
CMD /bin/bash
