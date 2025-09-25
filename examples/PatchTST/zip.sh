#!/bin/bash

# 상위 디렉토리 지정 (여기에 탐색을 시작할 경로를 입력하세요)
base_directory=$1 #"results"

# find 명령어로 모든 하위 디렉토리에서 pred.npy 파일을 찾습니다.
find "$base_directory" -type f -name "pred.npy" | while read file; do
    # 해당 파일의 디렉토리로 이동합니다.
    file_dir=$(dirname "$file")
    
    # pred.npy 파일을 압축하여 pred.zip으로 만듭니다.
    (cd "$file_dir" && zip -r pred.zip pred.npy)
    
    # 압축이 완료되면 pred.npy 파일을 삭제합니다.
    if [ $? -eq 0 ]; then
        rm "$file"
        echo "Compressed and removed: $file"
    else
        echo "Error compressing: $file"
    fi
done


# find 명령어로 모든 하위 디렉토리에서 pred.npy 파일을 찾습니다.
find "$base_directory" -type f -name "true.npy" | while read file; do
    # 해당 파일의 디렉토리로 이동합니다.
    file_dir=$(dirname "$file")
    
    # pred.npy 파일을 압축하여 pred.zip으로 만듭니다.
    (cd "$file_dir" && zip -r true.zip true.npy)
    
    # 압축이 완료되면 pred.npy 파일을 삭제합니다.
    if [ $? -eq 0 ]; then
        rm "$file"
        echo "Compressed and removed: $file"
    else
        echo "Error compressing: $file"
    fi
done
