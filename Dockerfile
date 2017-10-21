FROM alpine:3.6

WORKDIR /app

RUN apk update \
  && apk add \
    ca-certificates \
    libstdc++ \
    libgfortran \
    python3 \
    openblas \
    lapack \
  && apk add --virtual=build_dependencies \
    gfortran \
    g++ \
    make \
    openblas-dev \
    python3-dev

RUN python3 -m pip install --upgrade pip
RUN pip3 install numpy scipy scikit-learn

ADD src .

