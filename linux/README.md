# Linux

This benchmark times a Linux kernel build.

## Setup

    docker build docker -t benchmark-linux

## Run

    docker run -it --rm benchmark-linux
    make defconfig
    time make
