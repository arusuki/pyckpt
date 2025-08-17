#!/bin/bash

#
# 启动一个基于 Ray 的训练作业。
#
# 使用方法:
# ./run_ray.sh <model> <batch_size> <num_workers> <prefetch_factor> <iters> <job_id> <num_gpus> [其他参数]
#
# 示例:
# ./run_ray.sh resnet50 64 4 2 1000 job-001 8
#

if [ $# -lt 7 ]; then
    echo "用法: $0 model batch_size num_workers prefetch_factor iters job_id num_gpus [其他参数]"
    exit -1;
fi
 
export MODEL=$1
shift
export BATCH_SIZE=$1
shift
export NUM_WORKERS=$1
shift
export PREFETCH_FACTOR=$1
shift
export ITERS=$1
shift
export JOB_ID=$1
shift
export NUM_GPU=$1
shift

THIS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

judge_path="$THIS_DIR/datasets/imagenet"
TRAIN_FILE="$THIS_DIR/datasets/wikitext-2-raw/wiki.train.raw"

other_params="$@"

if [[ "$MODEL" == "bert" ]] || [[ "$MODEL" == "gpt2" ]]; then
    TRAIN_DIR=$TRAIN_FILE
else
    TRAIN_DIR=$judge_path
fi

# Hostfile 路径
# 脚本假设 Ray 集群已根据此 hostfile 配置并运行
hostfile=$THIS_DIR/hostfiles/hostfile
echo "使用 Hostfile: $hostfile (注意: Ray 集群的节点 rank 由 Ray 自身管理)"

# CUDA 配置
CUDA_FLAG=""
if [ $NUM_GPU -eq 0 ]; then
    CUDA_FLAG="--no-cuda"
fi

# Note:
# run.py 中的 ScalingConfig 和 DataLoader 都使用了 --num-workers 参数。
# 为了让 Ray Train 使用所有指定的 GPU，我们将 NUM_GPU 的值传递给 --num-workers。
# 这意味着 DataLoader 的 worker 数量也将被设置为 NUM_GPU 的值。
# 如果需要为 DataLoader 指定一个不同的 worker 数量，需要修改 run.py 以接受一个独立的参数。
# 此处，我们使用 NUM_WORKERS 参数来传递给 `--num-workers`，它应该等于您希望使用的 GPU 数量。

# 在 Ray 集群的头节点上执行 Python 训练脚本
# Ray Trainer 会自动将工作负载分发到集群中的工作节点上
exec python3 $THIS_DIR/run.py \
    --model $MODEL \
    --batch-size $BATCH_SIZE \
    --num-workers $NUM_WORKERS \
    --prefetch-factor $PREFETCH_FACTOR \
    --iters $ITERS \
    --train-dir $TRAIN_DIR \
    --job-id $JOB_ID \
    $CUDA_FLAG \
    $other_params