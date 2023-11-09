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
    touch "$dir_name/file$i.txt"
done
tar -cvf "$dir_name.tar" "$dir_name"
echo "파일 압축 완료"
mv "$dir_name.tar" "$dir_name"
cd "$dir_name" || exit
tar -xvf "$dir_name.tar"
echo "압축 해제 완료"
mv "$dir_name.tar" "$dir_name"
cd || exit
