#!/bin/sh
case "$2" in
  +)
    echo "$(($1 + $3))";;
  -)
    echo "$(($1 - $3))";;
  *)
    echo "정확히 입력해주세요";;
esac
