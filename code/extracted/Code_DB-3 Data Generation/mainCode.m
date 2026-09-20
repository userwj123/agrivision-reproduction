clc;
clear all;
close all;

% 1) Define Path and Folders for storing generated Data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
pn1 = uigetdir(pwd, 'Select Path for DB-3 folder');
backgroundFolder = fullfile(pn1, 'Images');
fruitFolder = fullfile(pn1, 'Fruit');
outputImageFolder = fullfile(pn1, 'Generated Images');
outputMaskFolder = fullfile(pn1, 'Generated Masks');
outputClutteredImageFolder = fullfile(pn1, 'Generated Images Cluttered');
outputClutteredMaskFolder = fullfile(pn1, 'Generated Masks Cluttered');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% 2) Adjustable Parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
numGeneratedImages = 5;             % Number of synthetic images to generate
berryCountRange = [10, 50];          % Range for number of blueberries per image
clutterPercentage = 0.3;             % Percentage of berries to clutter
clutterRatioRange = [0.2, 0.5];      % Range for clutter size (20-50% of patch)
maxBerrySizeFactor = 0.07;           % Max size of each berry relative to background
sizeVariationRange = [0.6, 0.9];     % Variation range for scaling blueberries
rotationRange = [0, 360];            % Range for random rotation angle
gaussianBlurRadius = 1.5;            % Gaussian blur radius for blueberry patches
edgeDilateRadius = 2;                % Radius for dilating edge mask
edgeBlurRadius = 2;                  % Gaussian blur radius for edge smoothing
leafDensityScale = 1.5;              % New parameter for scaling blueberry size based on leaf density
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% 3) Define green threshold for leaf detection (HSV color space)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
greenHueRange = [0.25, 0.45];        % Approximate hue range for green colors
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% 4) Get list of images and blueberry patches
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
backgroundImages = [dir(fullfile(backgroundFolder, '*.bmp'));dir(fullfile(backgroundFolder, '*.bmp'))];
fruitImages = [dir(fullfile(fruitFolder, '*.bmp'));dir(fullfile(fruitFolder, '*.bmp'))];
numGeneratedImages = numGeneratedImages * length(backgroundImages);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% 5) Loop to generate synthetic images
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for i = 1:numGeneratedImages
    % a) Generate a consistent base name for each sample
    baseName = sprintf('synthetic_image_%d', i);
    
    % b) Select random background image
    bgIndex = randi(length(backgroundImages));
    backgroundImage = imread(fullfile(backgroundFolder, backgroundImages(bgIndex).name));
    [bgHeight, bgWidth, ~] = size(backgroundImage);
    
    % c) Convert background image to HSV and isolate green areas
    hsvBackground = rgb2hsv(backgroundImage);
    greenMask = (hsvBackground(:,:,1) >= greenHueRange(1)) & (hsvBackground(:,:,1) <= greenHueRange(2)) & (hsvBackground(:,:,2) > 0.3);

    % d) Initialize masks for uncluttered and cluttered versions
    maskImage = zeros(bgHeight, bgWidth, 3, 'uint8');  % Uncluttered mask
    clutteredMaskImage = zeros(bgHeight, bgWidth, 3, 'uint8'); % Cluttered mask
    clutteredImage = backgroundImage;  % Duplicate background for cluttered image

    % e) Number of blueberries in this image
    numBerries = randi(berryCountRange);

    % f) Select a subset of patches to be cluttered (30% of total patches)
    clutteredIndices = randperm(numBerries, round(clutterPercentage * numBerries));

    % g) Loop to place each blueberry patch at the same position for both images
    for j = 1:numBerries
        % Load blueberry patch
        fruitIndex = randi(length(fruitImages));
        fruitImage = imread(fullfile(fruitFolder, fruitImages(fruitIndex).name));

        % Remove black background around blueberry
        fruitImage(fruitImage < 10) = 0;  % Assuming black background
        [fruitH, fruitW, ~] = size(fruitImage);
        
        % Adjust blueberry size based on leaf density
        maxBerrySize = maxBerrySizeFactor * min(bgHeight, bgWidth);
        greenDensity = sum(greenMask(:)) / numel(greenMask);  % Calculate density of green pixels
        scaleFactor = (sizeVariationRange(1) + rand() * (sizeVariationRange(2) - sizeVariationRange(1))) * ...
                      maxBerrySize * (1 + greenDensity * leafDensityScale) / max(fruitH, fruitW);
        fruitImage = imresize(fruitImage, scaleFactor);
        [fruitH, fruitW, ~] = size(fruitImage);

        % Apply random rotation and Gaussian blur
        rotationAngle = randi(rotationRange);
        fruitImage = imrotate(fruitImage, rotationAngle, 'bilinear', 'crop');
        fruitImage = imgaussfilt(fruitImage, gaussianBlurRadius);  % Slight blur for blending

        % Create circular mask
        centerX = round(fruitW / 2);
        centerY = round(fruitH / 2);
        radius = round(min(fruitH, fruitW) / 2);
        [X, Y] = meshgrid(1:fruitW, 1:fruitH);
        circularMask = (X - centerX).^2 + (Y - centerY).^2 <= radius^2;

        % Randomly position the blueberry patch on the green area of background
        validPositions = find(greenMask);
        if isempty(validPositions)
            continue; % Skip if no green area is available
        end
        randomPos = validPositions(randi(length(validPositions)));
        [y, x] = ind2sub([bgHeight, bgWidth], randomPos);

        % Ensure patch placement does not go out of bounds
        x = max(1, min(bgWidth - fruitW, x));
        y = max(1, min(bgHeight - fruitH, y));

        % Determine if this patch is to be cluttered
        isCluttered = ismember(j, clutteredIndices);
        
        % Create a cluttered version by removing a connected region if needed
        if isCluttered
            clutteredMask = circularMask;
            
            % Select a random side to remove (left, right, top, or bottom)
            removeSide = randi([1, 4]);
            clutterRatio = clutterRatioRange(1) + (clutterRatioRange(2) - clutterRatioRange(1)) * rand();
            switch removeSide
                case 1  % Left
                    clutteredMask(:, 1:round(clutterRatio * fruitW)) = 0;
                case 2  % Right
                    clutteredMask(:, end-round(clutterRatio * fruitW):end) = 0;
                case 3  % Top
                    clutteredMask(1:round(clutterRatio * fruitH), :) = 0;
                case 4  % Bottom
                    clutteredMask(end-round(clutterRatio * fruitH):end, :) = 0;
            end
        else
            clutteredMask = circularMask;
        end

        % Preserve contrast while blending into both uncluttered and cluttered images
        for row = 1:fruitH
            for col = 1:fruitW
                if circularMask(row, col) > 0.5
                    for c = 1:3
                        % Blend patch into both uncluttered and cluttered images
                        backgroundImage(y+row-1, x+col-1, c) = ...
                            uint8(0.9 * double(fruitImage(row, col, c)) + 0.1 * double(backgroundImage(y+row-1, x+col-1, c)));
                        
                        if clutteredMask(row, col) > 0.5
                            clutteredImage(y+row-1, x+col-1, c) = ...
                                uint8(0.9 * double(fruitImage(row, col, c)) + 0.1 * double(clutteredImage(y+row-1, x+col-1, c)));
                        end
                    end
                    % Update the mask images for both versions
                    maskImage(y+row-1, x+col-1, :) = 255;  % White in uncluttered mask
                    if clutteredMask(row, col) > 0.5
                        clutteredMaskImage(y+row-1, x+col-1, :) = 255; % White in cluttered mask
                    end
                end
            end
        end
    end

    % h) Apply boundary-based Gaussian smoothing on the edges of both masks
    edgeMask = edge(rgb2gray(maskImage), 'Canny');  % Detect boundaries
    edgeMaskDilated = imdilate(edgeMask, strel('disk', edgeDilateRadius));  % Dilate edges
    edgeMaskBlurred = imgaussfilt(double(edgeMaskDilated), edgeBlurRadius); % Gaussian blur on edges only

    % i) Blend edges into the background for uncluttered image
    for c = 1:3
        backgroundImage(:,:,c) = uint8(edgeMaskBlurred .* double(backgroundImage(:,:,c)) + (1 - edgeMaskBlurred) .* double(backgroundImage(:,:,c)));
    end

    % j) Blend edges into the background for cluttered image
    for c = 1:3
        clutteredImage(:,:,c) = uint8(edgeMaskBlurred .* double(clutteredImage(:,:,c)) + (1 - edgeMaskBlurred) .* double(clutteredImage(:,:,c)));
    end

    % k) Save uncluttered image and mask
    imwrite(backgroundImage, fullfile(outputImageFolder, sprintf('%s.bmp', baseName)));
    imwrite(maskImage, fullfile(outputMaskFolder, sprintf('%s.png', baseName)));

    % l) Save cluttered image and mask
    imwrite(clutteredImage, fullfile(outputClutteredImageFolder, sprintf('%s.bmp', baseName)));
    imwrite(clutteredMaskImage, fullfile(outputClutteredMaskFolder, sprintf('%s.png', baseName)));

    fprintf('Generated uncluttered and cluttered images and masks: %s\n', baseName);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%