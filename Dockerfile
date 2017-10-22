FROM alpine:edge

WORKDIR /app

RUN echo "http://dl-cdn.alpinelinux.org/alpine/edge/testing" >> /etc/apk/repositories

RUN apk update && \
  apk add \
    g++ \
    openblas-dev \
    py-numpy-dev \
    libpng-dev \
    freetype \
    freetype-dev \
    ca-certificates \
    python3 \
    py-scipy && \
  python3 -m pip install --upgrade pip && \
  pip3 install matplotlib && \
  pip3 install pandas && \
  pip3 install seaborn && \
  pip3 install scikit-learn && \
  apk del freetype-dev && \
  apk del py-numpy-dev && \
  apk del openblas-dev && \
  apk del g++

