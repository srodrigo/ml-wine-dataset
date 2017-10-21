FROM alpine:3.6

WORKDIR /app

RUN apk add --no-cache python3

ADD src .

