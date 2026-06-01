import utils.fflow as flw
import torch
import wandb

def main():
    # read options
    option = flw.read_option()
    print(option)
    if option['wandb']:
        # wandb expects authentication via `wandb login` CLI or WANDB_API_KEY env var.
        wandb.init(
            project='CondI',
            # name="{}_CW{:.2f}_CT{:.2f}_KL{:.2f}_P{:.2f}".format(option['task'], option['contrastive_weight'], option['temperature'], option['kl_weight'], option['proportion']),
            name=f"{option['model']}_{option['pm']}_{option['ps']}",
            group=option['task'],
            # group='ptbxl_reduce_missing',
            tags=[],
            config=option
            # ,
            # resume=True
        )
    # set random seed
    flw.setup_seed(option['seed'])
    # initialize server, clients and fedtask
    server = flw.initialize(option)
    # start federated optimization
    try:
        server.run()
    except Exception as e:
        # log the exception that happens during training-time
        print(e)
        flw.logger.exception("Exception Logged")
        raise RuntimeError

if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    torch.multiprocessing.set_sharing_strategy('file_system')
    main()