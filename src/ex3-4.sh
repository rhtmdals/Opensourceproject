#!/bin/sh
echo "리눅스가 재미있나요? (yes / no)"
read -r answer
case $answer in
  yes | y | Y | Yes | YES)
    echo "yes";;
  [nN]*)
    echo "no";;
  *)
    echo "yes 아니면 no로 입력해주세요";;
esac
