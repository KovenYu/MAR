from trainers import *
import time
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
import sys
sys.path.append("pycharm-debug-py3k.egg")
import pydevd
pydevd.settrace('172.18.218.16', port=8008, stdoutToServer=True, stderrToServer=True)


def main():
    opts = BaseOptions()
    args = opts.parse()
    logger = Logger(args.save_path)
    opts.print_options(logger)
    source_loader, target_loader, gallery_loader, probe_loader = \
        get_transfer_dataloaders(args.source, args.target, args.img_size,
                                 args.crop_size, args.padding, args.batch_size//2, False)
    args.num_classes = 4101

    if args.resume:
        trainer, start_epoch = load_checkpoint(args, logger)
    else:
        trainer = ReidTrainer(args, logger)
        start_epoch = 0

    total_epoch = args.epochs

    start_time = time.time()
    epoch_time = AverageMeter()

    for epoch in range(start_epoch, total_epoch):

        need_hour, need_mins, need_secs = convert_secs2time(epoch_time.avg * (total_epoch - epoch))
        need_time = 'Stage 1, [Need: {:02d}:{:02d}:{:02d}]'.format(need_hour, need_mins, need_secs)

        logger.print_log(
            '\n==>>{:s} [Epoch={:03d}/{:03d}] {:s}'.format(time_string(), epoch, total_epoch, need_time))

        meters_trn = trainer.train_epoch(source_loader, target_loader, epoch)
        logger.print_log('  **Train**  ' + create_stat_string(meters_trn))

        epoch_time.update(time.time() - start_time)
        start_time = time.time()

    meters_val = trainer.eval_performance(target_loader, gallery_loader, probe_loader)
    logger.print_log('  **Test**  ' + create_stat_string(meters_val))


if __name__ == '__main__':
    main()
