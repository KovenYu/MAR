from resnet import resnet50
from utils import *
import torch.nn as nn
import torch
import os
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR
from scipy.spatial.distance import cdist


class Trainer(object):
    def __init__(self):
        super(Trainer, self).__init__()

    def train(self, *names):
        """
        set the given attributes in names to the training state.
        if names is empty, call the train() method for all attributes which are instances of nn.Module.
        :param names:
        :return:
        """
        if not names:
            modules = []
            for attr_name in dir(self):
                attr = getattr(self, attr_name)
                if isinstance(attr, nn.Module):
                    modules.append(attr_name)
        else:
            modules = names

        for m in modules:
            getattr(self, m).train()

    def eval(self, *names):
        """
        set the given attributes in names to the evaluation state.
        if names is empty, call the eval() method for all attributes which are instances of nn.Module.
        :param names:
        :return:
        """
        if not names:
            modules = []
            for attr_name in dir(self):
                attr = getattr(self, attr_name)
                if isinstance(attr, nn.Module):
                    modules.append(attr_name)
        else:
            modules = names

        for m in modules:
            getattr(self, m).eval()


class ReidTrainer(Trainer):
    def __init__(self, args, logger):
        super(ReidTrainer, self).__init__()
        self.args = args
        self.logger = logger

        self.al_loss = nn.CrossEntropyLoss().cuda()
        self.rj_loss = JointLoss(args.margin).cuda()
        self.cml_loss = MultilabelLoss(args.batch_size).cuda()
        self.mdl_loss = DiscriminativeLoss(args.mining_ratio).cuda()

        self.net = resnet50(pretrained=False, num_classes=self.args.num_classes)
        if args.pretrain_path is None:
            self.logger.print_log('do not use pre-trained model. train from scratch.')
        elif os.path.isfile(args.pretrain_path):
            checkpoint = torch.load(args.pretrain_path)
            state_dict = parse_pretrained_checkpoint(checkpoint, args.num_classes)
            self.net.load_state_dict(state_dict, strict=False)
            self.logger.print_log('loaded pre-trained model from {}'.format(args.pretrain_path))
        else:
            self.logger.print_log('{} is not a file. train from scratch.'.format(args.pretrain_path))
        self.net = nn.DataParallel(self.net).cuda()

        bn_params, other_params = partition_params(self.net, 'bn')
        self.optimizer = torch.optim.SGD([{'params': bn_params, 'weight_decay': 0},
                                          {'params': other_params}], lr=args.lr, momentum=0.9, weight_decay=args.wd)
        self.lr_scheduler = MultiStepLR(self.optimizer, milestones=[int(args.epochs/8*5), int(args.epochs/8*7)])

        if 'Market' in args.target:
            N_target_samples = 12936
        elif 'Duke' in args.target:
            N_target_samples = 16522
        else:
            assert False, "Please modify the source code to specify" \
                          "the number of training samples :)"
        self.multilabel_memory = torch.zeros(N_target_samples, 4101)  # assume using MSMT17 as auxiliary dataset
        self.initialized = self.multilabel_memory.sum(dim=1) != 0

    def train_epoch(self, source_loader, target_loader, epoch):
        self.lr_scheduler.step()
        if not self.cml_loss.initialized or not self.mdl_loss.initialized:
            self.init_losses(target_loader)
        batch_time_meter = AverageMeter()
        stats = ('loss_source', 'loss_st', 'loss_ml', 'loss_target', 'loss_total')
        meters_trn = {stat: AverageMeter() for stat in stats}
        self.train()

        end = time.time()
        target_iter = iter(target_loader)
        for i, source_tuple in enumerate(source_loader):
            imgs = source_tuple[0].cuda()
            labels = source_tuple[1].cuda()

            try:
                target_tuple = next(target_iter)
            except StopIteration:
                target_iter = iter(target_loader)
                target_tuple = next(target_iter)
            imgs_target = target_tuple[0].cuda()
            labels_target = target_tuple[1].cuda()
            views_target = target_tuple[2].cuda()
            idx_target = target_tuple[3]

            features, similarity, _ = self.net(imgs)
            features_target, similarity_target, _ = self.net(imgs_target)
            scores = similarity * self.args.scala_ce
            loss_source = self.al_loss(scores, labels)
            agents = self.net.module.fc.weight.renorm(2, 0, 1e-5).mul(1e5)
            loss_st = self.rj_loss(features, agents.detach(), labels, similarity.detach(), features_target, similarity_target.detach())
            multilabels = F.softmax(features_target.mm(agents.detach().t_()*self.args.scala_ce), dim=1)
            loss_ml = self.cml_loss(torch.log(multilabels), views_target)
            if epoch < 1:
                loss_target = torch.Tensor([0]).cuda()
            else:
                multilabels_cpu = multilabels.detach().cpu()
                is_init_batch = self.initialized[idx_target]
                initialized_idx = idx_target[is_init_batch]
                uninitialized_idx = idx_target[~is_init_batch]
                self.multilabel_memory[uninitialized_idx] = multilabels_cpu[~is_init_batch]
                self.initialized[uninitialized_idx] = 1
                self.multilabel_memory[initialized_idx] = 0.9 * self.multilabel_memory[initialized_idx] \
                                                          + 0.1 * multilabels_cpu[is_init_batch]
                loss_target = self.mdl_loss(features_target, self.multilabel_memory[idx_target], labels_target)

            self.optimizer.zero_grad()
            loss_total = loss_target + self.args.lamb_1 * loss_ml + self.args.lamb_2 * \
                         (loss_source + self.args.beta * loss_st)
            loss_total.backward()
            self.optimizer.step()

            for k in stats:
                v = locals()[k]
                meters_trn[k].update(v.item(), self.args.batch_size)

            batch_time_meter.update(time.time() - end)
            freq = self.args.batch_size / batch_time_meter.avg
            end = time.time()
            if i % self.args.print_freq == 0:
                self.logger.print_log('  Iter: [{:03d}/{:03d}]   Freq {:.1f}   '.format(
                    i, len(source_loader), freq) + create_stat_string(meters_trn) + time_string())

        save_checkpoint(self, epoch, os.path.join(self.args.save_path, "checkpoints.pth"))
        return meters_trn

    def eval_performance(self, target_loader, gallery_loader, probe_loader):
        stats = ('r1', 'r5', 'r10', 'MAP')
        meters_val = {stat: AverageMeter() for stat in stats}
        self.eval()

        gallery_features, gallery_labels, gallery_views = extract_features(gallery_loader, self.net, index_feature=0)
        probe_features, probe_labels, probe_views = extract_features(probe_loader, self.net, index_feature=0)
        dist = cdist(gallery_features, probe_features, metric='cosine')
        CMC, MAP = eval_cmc_map(dist, gallery_labels, probe_labels, gallery_views, probe_views, ignore_MAP=False)
        r1 = CMC[0]
        r5 = CMC[4]
        r10 = CMC[9]

        for k in stats:
            v = locals()[k]
            meters_val[k].update(v.item(), self.args.batch_size)
        return meters_val

    def init_losses(self, target_loader):
        self.logger.print_log('initializing centers/threshold ...')
        if os.path.isfile(self.args.ml_path):
            (multilabels, views, pairwise_agreements) = torch.load(self.args.ml_path)
            self.logger.print_log('loaded ml from {}'.format(self.args.ml_path))
        else:
            self.logger.print_log('not found {}. computing ml...'.format(self.args.ml_path))
            sim, _, views = extract_features(target_loader, self.net, index_feature=1, return_numpy=False)
            multilabels = F.softmax(sim * self.args.scala_ce, dim=1)
            ml_np = multilabels.cpu().numpy()
            pairwise_agreements = 1 - pdist(ml_np, 'minkowski', p=1)/2
            self.logger.print_log('saving computed ml to {}'.format(self.args.ml_path))
            torch.save((multilabels, views, pairwise_agreements), self.args.ml_path)
        log_multilabels = torch.log(multilabels)
        self.cml_loss.init_centers(log_multilabels, views)
        self.logger.print_log('initializing centers done.')
        self.mdl_loss.init_threshold(pairwise_agreements)
        self.logger.print_log('initializing threshold done.')


def main():
    pass


if __name__ == '__main__':
    main()
