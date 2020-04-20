function pixelLabelColorbar(cmap, classNames)

% NOTE: This is sample code created by MATHWORKS and has been borrowed.
% Code Available @ https://www.mathworks.com/help/vision/examples/semantic-segmentation-using-deep-learning.html

% Add a colorbar to the current axis. The colorbar is formatted
% to display the class names with the color.

colormap(gca,cmap)

% Add colorbar to current figure.
c = colorbar('peer', gca);

% Use class names for tick marks.
c.TickLabels = classNames;
numClasses = size(cmap,1);

% Center tick labels.
c.Ticks = 1/(numClasses*2):1/numClasses:1;

% Remove tick mark.
c.TickLength = 0;
end