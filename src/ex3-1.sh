#!/bin/sh
read -r i
while [ $i -gt 0 ]
do
  echo "hello world"
  i=$((i-1))
done


