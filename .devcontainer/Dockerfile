# FROM mcr.microsoft.com/devcontainers/base:focal
FROM movesrwth/stormpy:1.7.0
RUN apt-get update

RUN apt-get -y install build-essential git cmake libboost-all-dev libcln-dev libgmp-dev libginac-dev automake libglpk-dev libhwloc-dev libz3-dev libxerces-c-dev libeigen3-dev

RUN apt-get update -qq && apt-get install -y --no-install-recommends \
    maven \
    uuid-dev \
    virtualenv

RUN apt-get -y install python3 python3-pip python3-dev
RUN apt-get -y install python-is-python3


WORKDIR /home/vscode/


ENTRYPOINT /bin/bash
