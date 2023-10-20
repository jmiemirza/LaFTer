#!/bin/bash
# custom config
DATA=data
TRAINER=clip_adapt
CFG=vit_b32
dset="$1"
method="$2"
CUDA_VISIBLE_DEVICES=1 python main.py \
--method ${method} \
--root ${DATA} \
--trainer ${TRAINER} \
--dataset-config-file configs/datasets/"${dset}".yaml \
--config-file configs/trainers/text_cls/${CFG}.yaml \
--output-dir output/${TRAINER}/${CFG}/"${dset}" \
--txt_epochs 100 \
--lr 0.001 \
--txt_cls 2
