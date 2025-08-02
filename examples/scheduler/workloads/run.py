import ray
import torch
import os
from ray.train.torch import TorchTrainer
from ray.train import ScalingConfig, RunConfig, CheckpointConfig
import ray.train as train

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from options import parser

def get_sargs(args):
    sargs = {"train_dir":args.train_dir, "model_name":args.model, "batch_size":args.batch_size, "iters":args.iters}
    sargs["num_workers"] = args.num_workers
    sargs["prefetch_factor"] = args.prefetch_factor
    
    return sargs

def train_loop_per_worker(config: dict):
    import sys
    import torch
    import time
    from ray.train import Checkpoint
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from options import parser
    from trainer import Trainer
    import util
    from nlp_model import NLPModel, get_nlp_dataset
    from cv_model import CVModel, get_cv_dataset
    
    args = config["args"]
    sargs = config["sargs"]
    model_name = sargs["model_name"]
    checkpoint_interval = config["checkpoint_interval"]
    
    rank = train.get_context().get_world_rank()

    trainer = None
    if rank == 0:
        print("Rank 0: Initializing custom Trainer for reporting to scheduler.")
        trainer = Trainer(
            args.scheduler_ip, 
            args.scheduler_port, 
            util.get_host_ip(),
            args.trainer_port, 
            args.job_id
        )

    if model_name in ['bert', 'gpt2']:
        train_dataset = get_nlp_dataset(args, sargs)
        model_instance = NLPModel(args, sargs)
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=sargs["batch_size"], shuffle=False,
            collate_fn=model_instance.collate, num_workers=sargs.get("num_workers", 2)
        )
    else:
        train_dataset = get_cv_dataset(sargs)
        model_instance = CVModel(args, sargs)
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=sargs["batch_size"], shuffle=False,
            num_workers=sargs.get("num_workers", 2)
        )
    
    train_dataloader = train.torch.prepare_data_loader(train_dataloader)

    model = train.torch.prepare_model(model_instance)
    optimizer, scheduler = model_instance.get_optimizer_and_scheduler(model, sargs)
    
    start_iter = 0
    checkpoint = train.get_checkpoint()
    if checkpoint:
        with checkpoint.as_directory() as checkpoint_dir:
            checkpoint_dict = torch.load(os.path.join(checkpoint_dir, "checkpoint.pt"))
            
            start_iter = checkpoint_dict["iter"] + 1
            model.load_state_dict(checkpoint_dict["model_state"])
            optimizer.load_state_dict(checkpoint_dict["optimizer_state"])
            if scheduler and "scheduler_state" in checkpoint_dict:
                scheduler.load_state_dict(checkpoint_dict["scheduler_state"])

    model.train()
    data_iter = iter(train_dataloader)
    
    cur_iter = 0
    time_all = 0.0
    itertime_list = []
    last_iter_for_reporting = 10
    
    cumulative_loss = 0.0
    
    for current_iter in range(start_iter, sargs['iters']):
        time_st = time.time()
        
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_dataloader)
            batch = next(data_iter)

        if model_name in ['bert', 'gpt2']:
            inputs, labels = batch
        else:
            inputs, labels = batch[0], batch[1]
        
        inputs = inputs.to(model.device)
        labels = labels.to(model.device)

        optimizer.zero_grad()
        loss = model_instance.forward_pass(model, inputs, labels)
        loss.backward()
        optimizer.step()
        if scheduler:
            scheduler.step()
        cumulative_loss += loss.item()

        if (current_iter + 1) % checkpoint_interval == 0:
            avg_loss_in_interval = cumulative_loss / checkpoint_interval
            
            checkpoint_dict = {
                "iter": current_iter,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
            }
            if scheduler:
                checkpoint_dict["scheduler_state"] = scheduler.state_dict()

            checkpoint = Checkpoint.from_dict(checkpoint_dict)
            
            train.report(
                {"loss": avg_loss_in_interval, "iter": current_iter}, 
                checkpoint=checkpoint
            )
            
            cumulative_loss = 0.0
            
        if rank == 0:
            time_end = time.time()
            iter_time = time_end - time_st
            
            # todo
            trainer.record(iter_time)
            
            print(f"Iter {cur_iter + 1}/{sargs['iters']}, Time: {iter_time:.4f}s, Loss: {loss.item():.4f}")
            
            if cur_iter + 1 > last_iter_for_reporting:
                time_all += iter_time
            
            if (cur_iter + 1) % 20 == 0 and cur_iter + 1 > last_iter_for_reporting:
                 avg_time = time_all / (cur_iter + 1 - last_iter_for_reporting)
                 itertime_list.append(avg_time)
                 print(f"Average iteration time since iter {last_iter_for_reporting}: {avg_time:.4f}s")

    if rank == 0:
        model_instance.print_info()
        trainer.report_itertime(itertime_list)


if __name__ == '__main__':
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    if not ray.is_initialized():
        ray.init(address="auto")

    sargs = get_sargs(args)
    checkpoint_interval = 100
    
    run_config = RunConfig(
        checkpoint_config=CheckpointConfig(
            num_to_keep=1,
            checkpoint_score_attribute="loss",
            checkpoint_score_order="min"
        ),
        name="training_experiment",
        storage_path="/tmp/ray_results"
    )
    
    trainer = TorchTrainer(
        train_loop_per_worker=train_loop_per_worker,
        train_loop_config={"args": args, "sargs": sargs, "checkpoint_interval": checkpoint_interval},
        scaling_config=ScalingConfig(
            num_workers=args.num_workers,
            use_gpu=args.cuda,
        ),
        run_config=run_config
    )
    
    result = trainer.fit()