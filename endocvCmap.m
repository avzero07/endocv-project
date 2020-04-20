%% EndoCV Colormap
% @author   - Akshay Viswakumar
% @email    - akshay.viswakumar@gmail.com
% @version  - v1.0
% @date     - 08-March-2020
%% Implementation

function cmap = endocvCmap()
% Defines a custom Colormap for the EndoCV Dataset. Used to color
% segmentation regions.

cmap = [
    1 1 0   % BE
    0 0 1   % Suspicious
    1 0 0   % HGD
    1 1 1   % Cancer
    0 1 0   % Polyp
    0 0 0   % Background
    ];
end