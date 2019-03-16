clear, clc
h = 384;
w = 128;

data = zeros(h, w, 3, 0, 'uint8');
labels = zeros(1, 0, 'int64');

[train_path, train_label] = textread('MSMT17_V1/list_train.txt','%s%n','delimiter', ' ');
 n = numel(train_label);
 data_t = zeros(h,w,3,n,'uint8');
 train_label = uint64(train_label);
 train_label = reshape(train_label, [1, n]); 
 for j = 1:n
        filepath = train_path{j};
        filepath = ['MSMT17_V1/train/', filepath];
        data_t(:, :, :, j) = uint8(imresize(imread(filepath), [h, w]));
        if mod(j, 1000) == 0
            fprintf('j == %d\n', j)
        end
            
 end
 data = cat(4, data, data_t);
 labels = cat(2, labels, train_label);
 
 
 [gallery_path, gallery_label] = textread('MSMT17_V1/list_gallery.txt','%s%n','delimiter', ' ');
 n = numel(gallery_label);
 data_t = zeros(h,w,3,n,'uint8');
 gallery_label = uint64(gallery_label);
 gallery_label = gallery_label+1041;
 gallery_label = reshape(gallery_label, [1, n]);
 for j = 1:n
        filepath = gallery_path{j};
        filepath = ['MSMT17_V1/test/', filepath];
        data_t(:, :, :, j) = uint8(imresize(imread(filepath), [h, w]));
        if mod(j, 1000) == 0 
            fprintf('j == %d\n', j)
        end
 end
 
 data = cat(4, data, data_t);
 labels = cat(2, labels, gallery_label);
 
 [query_path, query_label] = textread('MSMT17_V1/list_query.txt','%s%n','delimiter', ' ');
 n = numel(query_label);
 data_t = zeros(h,w,3,n,'uint8');
 query_label = uint64(query_label);
 query_label = query_label+1041;
 query_label = reshape(query_label, [1, n]);
 for j = 1:n
        filepath = query_path{j};
        filepath = ['MSMT17_V1/test/',  filepath];
        data_t(:, :, :, j) = uint8(imresize(imread(filepath), [h, w]));
        if mod(j, 1000) == 0
            fprintf('j == %d\n', j)
        end
 end
 
 data = cat(4, data, data_t);
 labels = cat(2, labels, query_label);

save('MSMT17.mat', 'data', 'labels', '-v7.3')