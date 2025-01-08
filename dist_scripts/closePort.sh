#!/bin/bash

if [ -z "$1" ]; then
  echo "Usage: $0 <port>"
  exit 1
fi

port="$1"

pid=$(lsof -t -i :$port)

if [ -z "$pid" ]; then
  echo "No process found using port $port"
else
  kill -9 $pid
  echo "Process with PID $pid terminated"
fi
