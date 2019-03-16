import os, sys, time
import numpy as np
import torch
import argparse
import torchvision.transforms as transforms

from ReIDdatasets import FullTraining, Market
import yaml
from tensorboardX import SummaryWriter
from scipy.spatial.distance import cdist, pdist
from collections import OrderedDict


class BaseOptions(object):
    """
    base options for deep learning for Re-ID.
    parse basic arguments by parse(), print all the arguments by print_options()
    """

    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.args = None

        self.parser.add_argument('--save_path', type=str, default='debug', help='Folder to save checkpoints and log.')
        self.parser.add_argument('--resume', default='', type=str, help='path to latest checkpoint (default: none)')
        self.parser.add_argument('--gpu', type=str, default='0', help='gpu used.')

    def parse(self):
        self.args = self.parser.parse_args()
        os.environ["CUDA_VISIBLE_DEVICES"] = self.args.gpu
        with open(os.path.join(self.args.save_path, 'args.yaml')) as f:
            extra_args = yaml.load(f)
        self.args = argparse.Namespace(**vars(self.args), **extra_args)
        return self.args

    def print_options(self, logger):
        logger.print_log("")
        logger.print_log("----- options -----".center(120, '-'))
        args = vars(self.args)
        string = ''
        for i, (k, v) in enumerate(sorted(args.items())):
            string += "{}: {}".format(k, v).center(40, ' ')
            if i % 3 == 2 or i == len(args.items()) - 1:
                logger.print_log(string)
                string = ''
        logger.print_log("".center(120, '-'))
        logger.print_log("")


class Logger(object):
    def __init__(self, save_path):
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        self.file = open(os.path.join(save_path, 'log_{}.txt'.format(time_string())), 'w')
        self.print_log("python version : {}".format(sys.version.replace('\n', ' ')))
        self.print_log("torch  version : {}".format(torch.__version__))

    def print_log(self, string):
        self.file.write("{}\n".format(string))
        self.file.flush()
        print(string)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class AgentLoss(torch.nn.Module):
    def __init__(self):
        super(AgentLoss, self).__init__()

    def forward(self, features, agents, labels):
        """
        :param features: shape=(BS, dim)
        :param agents: shape=(n_class, dim)
        :param labels: shape=(BS, dim)
        :return:
        """
        respective_agents = agents[labels]
        similarity_matrix = (features * respective_agents).sum(dim=1)
        loss = 1 - similarity_matrix.mean()
        return loss


class DiscriminativeLoss(torch.nn.Module):
    def __init__(self, mining_ratio=0.001):
        super(DiscriminativeLoss, self).__init__()
        self.mining_ratio = mining_ratio
        self.register_buffer('n_pos_pairs', torch.Tensor([0]))
        self.register_buffer('rate_TP', torch.Tensor([0]))
        self.moment = 0.1
        self.initialized = False

    def init_threshold(self, pairwise_agreements):
        pos = int(len(pairwise_agreements) * self.mining_ratio)
        sorted_agreements = np.sort(pairwise_agreements)
        t = sorted_agreements[-pos]
        self.register_buffer('threshold', torch.Tensor([t]).cuda())
        self.initialized = True

    def forward(self, features, multilabels, labels):
        """
        :param features: shape=(BS, dim)
        :param multilabels: (BS, n_class)
        :param labels: (BS,)
        :return:
        """
        P, N = self._partition_sets(features.detach(), multilabels, labels)
        if P is None:
            pos_exponant = torch.Tensor([1]).cuda()
            num = 0
        else:
            sdist_pos_pairs = []
            for (i, j) in zip(P[0], P[1]):
                sdist_pos_pair = (features[i] - features[j]).pow(2).sum()
                sdist_pos_pairs.append(sdist_pos_pair)
            pos_exponant = torch.exp(- torch.stack(sdist_pos_pairs)).mean()
            num = -torch.log(pos_exponant)
        if N is None:
            neg_exponant = torch.Tensor([0.5]).cuda()
        else:
            sdist_neg_pairs = []
            for (i, j) in zip(N[0], N[1]):
                sdist_neg_pair = (features[i] - features[j]).pow(2).sum()
                sdist_neg_pairs.append(sdist_neg_pair)
            neg_exponant = torch.exp(- torch.stack(sdist_neg_pairs)).mean()
        den = torch.log(pos_exponant + neg_exponant)
        loss = num + den
        return loss

    def _partition_sets(self, features, multilabels, labels):
        """
        partition the batch into confident positive, hard negative and others
        :param features: shape=(BS, dim)
        :param multilabels: shape=(BS, n_class)
        :param labels: shape=(BS,)
        :return:
        P: positive pair set. tuple of 2 np.array i and j.
            i contains smaller indices and j larger indices in the batch.
            if P is None, no positive pair found in this batch.
        N: negative pair set. similar to P, but will never be None.
        """
        f_np = features.cpu().numpy()
        ml_np = multilabels.cpu().numpy()
        p_dist = pdist(f_np)
        p_agree = 1 - pdist(ml_np, 'minkowski', p=1) / 2
        sorting_idx = np.argsort(p_dist)
        n_similar = int(len(p_dist) * self.mining_ratio)
        similar_idx = sorting_idx[:n_similar]
        is_positive = p_agree[similar_idx] > self.threshold.item()
        pos_idx = similar_idx[is_positive]
        neg_idx = similar_idx[~is_positive]
        P = dist_idx_to_pair_idx(len(f_np), pos_idx)
        N = dist_idx_to_pair_idx(len(f_np), neg_idx)
        self._update_threshold(p_agree)
        self._update_buffers(P, labels)
        return P, N

    def _update_threshold(self, pairwise_agreements):
        pos = int(len(pairwise_agreements) * self.mining_ratio)
        sorted_agreements = np.sort(pairwise_agreements)
        t = torch.Tensor([sorted_agreements[-pos]]).cuda()
        self.threshold = self.threshold * (1 - self.moment) + t * self.moment

    def _update_buffers(self, P, labels):
        if P is None:
            self.n_pos_pairs = 0.9 * self.n_pos_pairs
            return 0
        n_pos_pairs = len(P[0])
        count = 0
        for (i, j) in zip(P[0], P[1]):
            count += labels[i] == labels[j]
        rate_TP = float(count) / n_pos_pairs
        self.n_pos_pairs = 0.9 * self.n_pos_pairs + 0.1 * n_pos_pairs
        self.rate_TP = 0.9 * self.rate_TP + 0.1 * rate_TP


class JointLoss(torch.nn.Module):
    def __init__(self, margin=1):
        super(JointLoss, self).__init__()
        self.margin = margin
        self.sim_margin = 1 - margin / 2

    def forward(self, features, agents, labels, similarity, features_target, similarity_target):
        """
        :param features: shape=(BS/2, dim)
        :param agents: shape=(n_class, dim)
        :param labels: shape=(BS/2,)
        :param features_target: shape=(BS/2, n_class)
        :return:
        """
        loss_terms = []
        arange = torch.arange(len(agents)).cuda()
        zero = torch.Tensor([0]).cuda()
        for (f, l, s) in zip(features, labels, similarity):
            loss_pos = (f - agents[l]).pow(2).sum()
            loss_terms.append(loss_pos)
            neg_idx = arange != l
            hard_agent_idx = neg_idx & (s > self.sim_margin)
            if torch.any(hard_agent_idx):
                hard_neg_sdist = (f - agents[hard_agent_idx]).pow(2).sum(dim=1)
                loss_neg = torch.max(zero, self.margin - hard_neg_sdist).mean()
                loss_terms.append(loss_neg)
        for (f, s) in zip(features_target, similarity_target):
            hard_agent_idx = s > self.sim_margin
            if torch.any(hard_agent_idx):
                hard_neg_sdist = (f - agents[hard_agent_idx]).pow(2).sum(dim=1)
                loss_neg = torch.max(zero, self.margin - hard_neg_sdist).mean()
                loss_terms.append(loss_neg)
        loss_total = torch.mean(torch.stack(loss_terms))
        return loss_total


class MultilabelLoss(torch.nn.Module):
    def __init__(self, batch_size, use_std=True):
        super(MultilabelLoss, self).__init__()
        self.use_std = use_std
        self.moment = batch_size / 10000
        self.initialized = False

    def init_centers(self, log_multilabels, views):
        """
        :param log_multilabels: shape=(N, n_class)
        :param views: (N,)
        :return:
        """
        univiews = torch.unique(views)
        mean_ml = []
        std_ml = []
        for v in univiews:
            ml_in_v = log_multilabels[views == v]
            mean = ml_in_v.mean(dim=0)
            std = ml_in_v.std(dim=0)
            mean_ml.append(mean)
            std_ml.append(std)
        center_mean = torch.mean(torch.stack(mean_ml), dim=0)
        center_std = torch.mean(torch.stack(std_ml), dim=0)
        self.register_buffer('center_mean', center_mean)
        self.register_buffer('center_std', center_std)
        self.initialized = True

    def _update_centers(self, log_multilabels, views):
        """
        :param log_multilabels: shape=(BS, n_class)
        :param views: shape=(BS,)
        :return:
        """
        univiews = torch.unique(views)
        means = []
        stds = []
        for v in univiews:
            ml_in_v = log_multilabels[views == v]
            if len(ml_in_v) == 1:
                continue
            mean = ml_in_v.mean(dim=0)
            means.append(mean)
            if self.use_std:
                std = ml_in_v.std(dim=0)
                stds.append(std)
        new_mean = torch.mean(torch.stack(means), dim=0)
        self.center_mean = self.center_mean * (1 - self.moment) + new_mean * self.moment
        if self.use_std:
            new_std = torch.mean(torch.stack(stds), dim=0)
            self.center_std = self.center_std * (1 - self.moment) + new_std * self.moment

    def forward(self, log_multilabels, views):
        """
        :param log_multilabels: shape=(BS, n_class)
        :param views: shape=(BS,)
        :return:
        """
        self._update_centers(log_multilabels.detach(), views)

        univiews = torch.unique(views)
        loss_terms = []
        for v in univiews:
            ml_in_v = log_multilabels[views == v]
            if len(ml_in_v) == 1:
                continue
            mean = ml_in_v.mean(dim=0)
            loss_mean = (mean - self.center_mean).pow(2).sum()
            loss_terms.append(loss_mean)
            if self.use_std:
                std = ml_in_v.std(dim=0)
                loss_std = (std - self.center_std).pow(2).sum()
                loss_terms.append(loss_std)
        loss_total = torch.mean(torch.stack(loss_terms))
        return loss_total


def time_string():
    ISOTIMEFORMAT = '%Y-%m-%d %X'
    string = '[{}]'.format(time.strftime(ISOTIMEFORMAT, time.localtime(time.time())))
    return string


def extract_features(loader, model, index_feature=None, return_numpy=True):
    """
    extract features for the given loader using the given model
    if loader.dataset.require_views is False, the returned 'views' are empty.
    :param loader: a ReIDDataset that has attribute require_views
    :param model: returns a tuple containing the feature or only return the feature. if latter, index_feature be None
    model can also be a tuple of nn.Module, indicating that the feature extraction is multi-stage.
    in this case, index_feature should be a tuple of the same size.
    :param index_feature: in the tuple returned by model, the index of the feature.
    if the model only returns feature, this should be set to None.
    :param return_numpy: if True, return numpy array; otherwise return torch tensor
    :return: features, labels, views, np array
    """
    if type(model) is not tuple:
        models = (model,)
        indices_feature = (index_feature,)
    else:
        assert len(model) == len(index_feature)
        models = model
        indices_feature = index_feature
    for m in models:
        m.eval()

    labels = []
    views = []
    features = []

    require_views = loader.dataset.require_views
    for i, data in enumerate(loader):
        imgs = data[0].cuda()
        label_batch = data[1]
        inputs = imgs
        for m, feat_idx in zip(models, indices_feature):
            with torch.no_grad():
                output_tuple = m(inputs)
            feature_batch = output_tuple if feat_idx is None else output_tuple[feat_idx]
            inputs = feature_batch

        features.append(feature_batch)
        labels.append(label_batch)
        if require_views:
            view_batch = data[2]
            views.append(view_batch)
    features = torch.cat(features, dim=0)
    labels = torch.cat(labels, dim=0)
    views = torch.cat(views, dim=0) if require_views else views
    if return_numpy:
        return np.array(features.cpu()), np.array(labels.cpu()), np.array(views.cpu())
    else:
        return features, labels, views


def create_stat_string(meters):
    stat_string = ''
    for stat, meter in meters.items():
        stat_string += '{} {:.3f}   '.format(stat, meter.avg)
    return stat_string


def convert_secs2time(epoch_time):
    need_hour = int(epoch_time / 3600)
    need_mins = int((epoch_time - 3600 * need_hour) / 60)
    need_secs = int(epoch_time - 3600 * need_hour - 60 * need_mins)
    return need_hour, need_mins, need_secs


def eval_cmc_map(dist, gallery_labels, probe_labels, gallery_views=None,
                 probe_views=None, ignore_MAP=True):
    """
    :param dist: 2-d np array, shape=(num_gallery, num_probe), distance matrix.
    :param gallery_labels: np array, shape=(num_gallery,)
    :param probe_labels:
    :param gallery_views: np array, shape=(num_gallery,) if specified, for any probe image,
    the gallery correct matches from the same view are ignored.
    :param probe_views: must be specified if gallery_views are specified.
    :param ignore_MAP: is True, only compute cmc
    :return:
    CMC: np array, shape=(num_gallery,). Measured by percentage
    MAP: np array, shape=(1,). Measured by percentage
    """
    gallery_labels = np.asarray(gallery_labels)
    probe_labels = np.asarray(probe_labels)
    dist = np.asarray(dist)

    is_view_sensitive = False
    num_gallery = gallery_labels.shape[0]
    num_probe = probe_labels.shape[0]
    if gallery_views is not None or probe_views is not None:
        assert gallery_views is not None and probe_views is not None, \
            'gallery_views and probe_views must be specified together. \n'
        gallery_views = np.asarray(gallery_views)
        probe_views = np.asarray(probe_views)
        is_view_sensitive = True
    cmc = np.zeros((num_gallery, num_probe))
    ap = np.zeros((num_probe,))
    for i in range(num_probe):
        cmc_ = np.zeros((num_gallery,))
        dist_ = dist[:, i]
        probe_label = probe_labels[i]
        gallery_labels_ = gallery_labels
        if is_view_sensitive:
            probe_view = probe_views[i]
            is_from_same_view = gallery_views == probe_view
            is_correct = gallery_labels == probe_label
            should_be_excluded = is_from_same_view & is_correct
            dist_ = dist_[~should_be_excluded]
            gallery_labels_ = gallery_labels_[~should_be_excluded]
        ranking_list = np.argsort(dist_)
        inference_list = gallery_labels_[ranking_list]
        positions_correct_tuple = np.nonzero(probe_label == inference_list)
        positions_correct = positions_correct_tuple[0]
        pos_first_correct = positions_correct[0]
        cmc_[pos_first_correct:] = 1
        cmc[:, i] = cmc_

        if not ignore_MAP:
            num_correct = positions_correct.shape[0]
            for j in range(num_correct):
                last_precision = float(j) / float(positions_correct[j]) if j != 0 else 1.0
                current_precision = float(j + 1) / float(positions_correct[j] + 1)
                ap[i] += (last_precision + current_precision) / 2.0 / float(num_correct)

    CMC = np.mean(cmc, axis=1)
    MAP = np.mean(ap)
    return CMC * 100, MAP * 100


def save_checkpoint(trainer, epoch, save_path, is_best=False):
    logger = trainer.logger
    recorder = trainer.recorder
    trainer.logger = None
    trainer.recorder = None
    if not os.path.isdir(os.path.dirname(save_path)):
        os.mkdir(os.path.dirname(save_path))
    torch.save((trainer, epoch), save_path)
    if is_best:
        best_path = save_path + '.best'
        torch.save((trainer, epoch), best_path)
    trainer.logger = logger
    trainer.recorder = recorder


def load_checkpoint(args, logger):
    """
    load a checkpoint (containing a trainer and an epoch number) and assign a logger to the loaded trainer.
    the differences between the loaded trainer.args and input args would be print to logger.
    :param args:
    :param logger:
    :return:
    """
    load_path = args.resume
    assert os.path.isfile(load_path)
    logger.print_log("=> loading checkpoint '{}'".format(load_path))
    (trainer, epoch) = torch.load(load_path)
    trainer.logger = logger
    trainer.recorder = SummaryWriter(os.path.join(args.save_path, 'tb_logs'))

    old_args = trainer.args
    trainer.args = args

    attributes = vars(args)
    old_attributes = vars(old_args)
    for name, value in attributes.items():
        if name == 'resume' or name == 'gpu':
            continue
        if name in old_attributes:
            old_value = old_attributes[name]
            if old_value != value:
                logger.print_log(
                    "args.{} was {} but now is replaced by the newly specified one: {}.".format(name, old_value,
                                                                                                value))
        else:
            logger.print_log("args.{} was not specified in the checkpoint.".format(name))
    return trainer, epoch


def partition_params(module, strategy, *desired_modules):
    """
    partition params into desired part and the residual
    :param module:
    :param strategy: choices are: ['bn', 'specified'].
    'bn': desired_params = bn_params
    'specified': desired_params = all params within desired_modules
    :param desired_modules: strings, each corresponds to a specific module
    :return: two lists
    """
    if strategy == 'bn':
        desired_params_set = set()
        for m in module.modules():
            if (isinstance(m, torch.nn.BatchNorm1d) or
                    isinstance(m, torch.nn.BatchNorm2d) or
                    isinstance(m, torch.nn.BatchNorm3d)):
                desired_params_set.update(set(m.parameters()))
    elif strategy == 'specified':
        desired_params_set = set()
        for module_name in desired_modules:
            sub_module = module.__getattr__(module_name)
            for m in sub_module.modules():
                desired_params_set.update(set(m.parameters()))
    else:
        assert False, 'unknown strategy: {}'.format(strategy)
    all_params_set = set(module.parameters())
    other_params_set = all_params_set.difference(desired_params_set)
    desired_params = list(desired_params_set)
    other_params = list(other_params_set)
    return desired_params, other_params


def get_transfer_dataloaders(source, target, img_size, crop_size, padding, batch_size, use_source_stat=True):
    """
    get source/target/gallery/probe dataloaders, where
    source loader is FullTraining, others are Market.
    :return: the four dataloaders
    """

    source_data = FullTraining('data/{}.mat'.format(source))
    target_data = Market('data/{}.mat'.format(target), state='train')
    gallery_data = Market('data/{}.mat'.format(target), state='gallery')
    probe_data = Market('data/{}.mat'.format(target), state='probe')

    mean = np.array([0.485, 0.406, 0.456])
    std = np.array([0.229, 0.224, 0.225])
    if not use_source_stat:
        mean = source_data.return_mean() / 255.0
        std = source_data.return_std() / 255.0

    source_transform = transforms.Compose(
        [transforms.RandomHorizontalFlip(), transforms.Resize(img_size),
         transforms.RandomCrop(crop_size, padding), transforms.ToTensor(), transforms.Normalize(mean, std)])
    if not use_source_stat:
        mean = target_data.return_mean() / 255.0
        std = target_data.return_std() / 255.0
    target_transform = transforms.Compose(
        [transforms.RandomHorizontalFlip(), transforms.Resize(img_size),
         transforms.RandomCrop(crop_size, padding), transforms.ToTensor(), transforms.Normalize(mean, std)])
    test_transform = transforms.Compose(
        [transforms.Resize(img_size), transforms.ToTensor(), transforms.Normalize(mean, std)])

    source_data.turn_on_transform(transform=source_transform)
    target_data.turn_on_transform(transform=target_transform)
    gallery_data.turn_on_transform(transform=test_transform)
    probe_data.turn_on_transform(transform=test_transform)

    n_workers = 1 if 'MSMT' in source else 2

    source_loader = torch.utils.data.DataLoader(source_data, batch_size=batch_size, shuffle=True,
                                                num_workers=n_workers, pin_memory=True, drop_last=True)
    target_loader = torch.utils.data.DataLoader(target_data, batch_size=batch_size, shuffle=True,
                                                num_workers=n_workers, pin_memory=True, drop_last=True)
    gallery_loader = torch.utils.data.DataLoader(gallery_data, batch_size=batch_size, shuffle=False,
                                                 num_workers=n_workers, pin_memory=True)
    probe_loader = torch.utils.data.DataLoader(probe_data, batch_size=batch_size, shuffle=False,
                                               num_workers=n_workers, pin_memory=True)

    return source_loader, target_loader, gallery_loader, probe_loader


def construct_pairwise_similarity_matrix(X, Y, sigma2=1, metric='euclidean'):
    """ construct a similarity matrix M, where Mij is the similarity between Xi and Yj
    defined as exp(-dist(Xi, Yj)^2/sigma2)
    :param X: 2d tensor/array, shape=(num_instances, dimension)
    :param Y: 2d tensor/array, shape=(num_instances, dimension)
    :param sigma2: scale factor in gaussian kernel
    :param metric: the metric used for computing the distances
    :return:
    """
    if len(X.shape) == 1:
        X = [X]
    if len(Y.shape) == 1:
        Y = [Y]
    X = np.array(X)
    Y = np.array(Y)
    dist = cdist(X, Y, metric=metric)
    similarity_matrix = np.exp(-dist ** 2 / sigma2)
    return similarity_matrix


def parse_pretrained_checkpoint(checkpoint, num_classes, fc_layer_name='fc'):
    """
    :param checkpoint: OrderedDict (state_dict) or a tuple (checkpoint)
    :param num_classes:
    :param fc_layer_name:
    :return: state_dict: a state dict whose fc layer is processed,
    i.e. if the fc output is not num_classes, remove the fc weight and fc bias (if exists)
    """
    if isinstance(checkpoint, OrderedDict):
        print('loaded a state dict.\n')
        state_dict = checkpoint
    elif isinstance(checkpoint, tuple):
        print('loaded a checkpoint.\n')
        net = checkpoint[0].net
        if isinstance(net, torch.nn.DataParallel):
            state_dict = net.module.state_dict()
        else:
            state_dict = net.state_dict()
    else:
        assert False, 'unknown type {}\n'.format(type(checkpoint))
    fc_weight_name = '{}.weight'.format(fc_layer_name)
    fc_weight = state_dict[fc_weight_name]
    output_dim = fc_weight.shape[0]
    if output_dim != num_classes:
        print('The output dim not match the specified num_classes. fc param is removed.\n')
        state_dict.pop(fc_weight_name)
        if '{}.bias'.format(fc_layer_name) in state_dict:
            state_dict.pop('{}.bias'.format(fc_layer_name))
    return state_dict


def pair_idx_to_dist_idx(d, i, j):
    """
    :param d: numer of elements
    :param i: np.array. i < j in every element
    :param j: np.array
    :return:
    """
    assert np.sum(i < j) == len(i)
    index = d * i - i * (i + 1) / 2 + j - 1 - i
    return index.astype(int)


def dist_idx_to_pair_idx(d, i):
    """
    :param d: number of samples
    :param i: np.array
    :return:
    """
    if i.size == 0:
        return None
    b = 1 - 2 * d
    x = np.floor((-b - np.sqrt(b ** 2 - 8 * i)) / 2).astype(int)
    y = (i + x * (b + x + 2) / 2 + 1).astype(int)
    return x, y


def test():
    pass


if __name__ == '__main__':
    test()
