function uniquized_labels = uniquize( labels )

uniquized_labels = labels;
% set the range into 1,2,3,...,n
uni = unique(labels);
for i = 1:numel(uni)
    uniquized_labels(labels == uni(i)) = i;
end

end

