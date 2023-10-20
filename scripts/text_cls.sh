#!/bin/bash
# custom config
DATA=data
TRAINER=clip_txt_cls
CFG=vit_b32
dset="$1"
method="$2"
txt_cls="$3"
CUDA_VISIBLE_DEVICES=1 python main.py \
--method "${method}" \
--root ${DATA} \
--trainer ${TRAINER} \
--dataset-config-file configs/datasets/"${dset}".yaml \
--config-file configs/trainers/text_cls/${CFG}.yaml \
--output-dir output/${TRAINER}/${CFG}/"${dset}" \
--txt_epochs 5000 \
--lr 0.001 \
--txt_cls "${txt_cls}"
