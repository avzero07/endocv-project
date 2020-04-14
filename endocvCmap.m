function cmap = endocvCmap()
% Define the colormap used by EndoCV dataset.

cmap = [
    128 128 0     % Tree
    192 128 128   % SignSymbol
    64 64 128     % Fence
    64 0 128      % Car
    32 32 32       % Pedestrian
    0 128 192     % Bicyclist
    ];


% Normalize between [0 1].
cmap = cmap ./ 255;
end