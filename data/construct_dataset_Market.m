clear, clc
h = 384;
w = 128;
data = zeros(h, w, 3, 0, 'uint8');
idxViews = zeros(1, 0, 'int64');
labels = zeros(1, 0, 'int64');
set =zeros(1, 0, 'uint8');

folder = {'train', 'test', 'query'};
for i = 1:numel(folder)
    path = ['Market-1501-v15.09.15/', folder{i}];
    directory = dir(path);
    idx = arrayfun(@(x)length(x.name)<5||~strcmp(x.name(5), '_'), directory);
    directory = directory(~idx);
    n = numel(directory);
    data_t = zeros(h,w,3,n,'uint8');
    labels_t = zeros(1, n, 'int64');
    idxViews_t = zeros(1, n, 'int64');
    set_t = ones(1, n, 'uint8');
    for j = 1:n
        filename = directory(j).name;
        labels_t(j) = uint64(str2double(filename(1:4)));
        idxViews_t(j) = uint64(str2double(filename(7)));
        filepath = [path, '/', filename];
        data_t(:, :, :, j) = uint8(imresize(imread(filepath), [h, w]));
        set_t(j) = i;
        if mod(j, 1000) == 0
            fprintf('j == %d\n', j)
        end
    end
    if i > 1 % test(gal, 3) or query(prb, 4)
        set_t = set_t + 1;
    end
    data = cat(4, data, data_t);
    labels = cat(2, labels, labels_t);
    set = cat(2, set, set_t);
    idxViews = cat(2, idxViews, idxViews_t);
end

train_data = data(:, :, :, set == 1);
train_labels = labels(set == 1);
train_views = idxViews(set == 1);
gallery_data = data(:, :, :, set == 3);
gallery_labels = labels(set == 3);
gallery_views = idxViews(set == 3);
probe_data = data(:, :, :, set == 4);
probe_labels = labels(set == 4);
probe_views = idxViews(set == 4);

train_labels = uniquize(train_labels);
gallery_labels = uniquize(gallery_labels);
probe_labels = uniquize(probe_labels);

gallery_labels = gallery_labels - 1;
train_labels = train_labels - 1;

save('Market.mat', 'gallery_data', 'gallery_labels', ...
    'probe_data', 'probe_labels', 'train_data', 'train_labels', ...
    'gallery_views', 'probe_views', 'train_views', '-v7.3')