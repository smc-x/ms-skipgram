#!/usr/bin/env bash

atc --model=../model/skipgram.air \
    --framework=1 \
    --output=../model/skipgram \
    --soc_version=Ascend310 