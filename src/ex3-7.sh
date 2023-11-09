#!/bin/bash
dir_name=$1
if [ ! -d "$dir_name" ]; then
    mkdir "$dir_name"
    echo "폴더 '$dir_name' 생성 완료"
else
    echo "폴더 '$dir_name' 이미 존재"
fi
for i in 0 1 2 3 4
do
    mkdir "$dir_name/file$i"
    touch "$dir_name/file$i.txt"
    ln -s "$dir_name/file$i.txt" "$dir_name/file$i/file$i.txt"
done
