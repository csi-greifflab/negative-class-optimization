#!/bin/bash

# glam2
docker run -it -v `pwd`:/home/meme memesuite/memesuite glam2 -h
docker run -it -v `pwd`:/home/meme memesuite/memesuite glam2 -o data/MiniAbsolut/1ADQ/glam2/high_test_5000.glam2 p data/MiniAbsolut/1ADQ/fasta/high_test_5000.fasta

# xstreme example
docker run -it -v `pwd`:/home/meme memesuite/memesuite xstreme \
    --p data/MiniAbsolut/1ADQ/fasta/high_test_5000.fasta \
    -oc data/MiniAbsolut/1ADQ/xstreme/high_test_5000 \
    --protein \
    --minw 4 \
    --maxw 11
    # -- n  # control sequences in fasta
