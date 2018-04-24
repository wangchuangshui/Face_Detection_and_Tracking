function videoout=face(video)
% create a detection for face 
faceDetector = vision.CascadeObjectDetector('ClassificationModel','FrontalFaceCART');
% read video frames and  perform face detection of the frames
videoFileReader = vision.VideoFileReader(video);
videoFrame      = step(videoFileReader);
bbox            = step(faceDetector, videoFrame);
% Draw the detected face
videoFrame = insertShape(videoFrame, 'Rectangle', bbox);
%imshow(videoFrame);
 for i=1:size(bbox,1)
% Convert the box to a list of 4 points
bboxPoints{1,i}= bbox2points(bbox(i,:));
I{1,i}=rgb2gray(imcrop(videoFrame,bbox(i,:)));
text{1,i}=facere(I{1,i});
text1=['Face Number Is ' num2str(size(bbox,1))];
% Detect The Feature Points Of face
points{1,i} = detectMinEigenFeatures(rgb2gray(videoFrame), 'ROI', bbox(i,:));
% create a tracker
pointTracker{1,i} = vision.PointTracker('MaxBidirectionalError',i);
%Initialize tracking with initial point position and initial point
 points{1,i}= points{1,i}.Location;
initialize(pointTracker{1,i},points{1,i}, videoFrame);
end
videoPlayer  = vision.VideoPlayer('Position',...
    [0 0 [size(videoFrame, 2), size(videoFrame, 1)]]);
% Conversion between the points in the previous frame and the current frame
for i=1:size(bbox,1)
oldPoints{1,i} = points{1,i};
end
while ~isDone(videoFileReader)
    % Get the next frame
    videoFrame = step(videoFileReader);
    % track the points
    for i=1:size(bbox,1)
   [points{1,i}, isFound{1,i}] = step(pointTracker{1,i}, videoFrame);
    visiblePoints{1,i} = points{1,i}(isFound{1,i}, :);
    oldInliers{1,i} = oldPoints{1,i}(isFound{1,i}, :);
   if size(visiblePoints{1,i}, 1) >= 2 
        [xform{1,i}, oldInliers{1,i}, visiblePoints{1,i}] = estimateGeometricTransform(...
            oldInliers{1,i}, visiblePoints{1,i}, 'similarity', 'MaxDistance', 4);
        bboxPoints{1,i} = transformPointsForward(xform{1,i}, bboxPoints{1,i});
        % Insert the tracking box
        bboxPolygon{1,i} = reshape(bboxPoints{1,i}', 1, []);
        videoFrame = insertShape(videoFrame, 'Polygon', bboxPolygon{1,i}, ...
            'LineWidth', 2);
        %Insert the text
    videoFrame = insertText(videoFrame,[ bboxPolygon{1,i}(1,3) bboxPolygon{1,i}(1,4)],text{1,i});
    videoFrame = insertText(videoFrame,[ 100 100],text1,'FontSize',30,'TextColor','red');
        % Show the tracking points
%        videoFrame = insertMarker(videoFrame, visiblePoints{1,i}, 'star', ...
%             'Color', 'white');
       % Reset these points
       oldPoints{1,i} = visiblePoints{1,i};
       setPoints(pointTracker{1,i}, oldPoints{1,i});  
   end
end
    % Play the processed video
    step(videoPlayer, videoFrame);
end
% clear
release(videoFileReader);
release(videoPlayer);
release(pointTracker{1,i});
end