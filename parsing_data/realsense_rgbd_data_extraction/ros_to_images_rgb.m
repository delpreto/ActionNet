
bag_filename = 'kitchen_depth-raw_2022-06-07-17-31-35.bag';
topic_data = '/kitchen_depth/color/image_raw';
start_offset_s = 100; % nan to start from beginning
duration_s = 10;      % nan to use whole file

fprintf('\n==========================\n');

%% Load the ROS bag.
fprintf('Loading the ROS bag... ');

% Load the bag file.
bag = rosbag(bag_filename);
% Select the desired times.
if isnan(start_offset_s)
    start_offset_s = 0;
end
if isnan(duration_s)
    duration_s = inf;
end
start_time_s = bag.StartTime + start_offset_s;
end_time_s = bag.StartTime + start_offset_s + duration_s;
bag = select(bag, 'Time', [start_time_s, end_time_s], 'Topic', {topic_data});

% Separate the data from the other topics.
data_bag = select(bag, 'Topic', topic_data);

% Get some metadata and printouts.
num_frames = data_bag.NumMessages;
duration_s = bag.EndTime - bag.StartTime;

fprintf('done\n');
bag_topics = bag.AvailableTopics
fprintf('Stream length: %0.2f minutes (%d frames)\n', duration_s/60, num_frames);
fprintf('Frame rate: %0.2f Hz\n', (num_frames-1)/duration_s);
fprintf('\n');

%% Extract each message as a structure.
fprintf('Creating image structs from the messages... ');
imgs_struct = readMessages(bag, 'DataFormat', 'struct');
fprintf('done\n');

%% Convert to images.
fprintf('Extracting images... ');
imgs = uint8(nan(num_frames, imgs_struct{1}.Height, imgs_struct{1}.Width, 3));
imgs_t = nan(num_frames, 1);
for frame_index = 1:num_frames
    if mod(frame_index, round(length(imgs_struct)/100)) == 0
        fprintf('\n  Processed %05d/%d frames (%0.1f%%) ', (frame_index-1), num_frames, 100*(frame_index-1)/num_frames);
    end
    imgs(frame_index, :, :, :) = rosReadImage(imgs_struct{frame_index}, 'PreserveStructureOnRead', false);
    % Note that the timestamps are stored as integers, so need to convert
    %  to doubles before dividing to convert nanoseconds to seconds.
    t_s = double(imgs_struct{frame_index}.Header.Stamp.Sec);
    t_ns = double(imgs_struct{frame_index}.Header.Stamp.Nsec);
    imgs_t(frame_index) = t_s + t_ns/1e9;
end
fprintf('done\n');

%% Show the frames
figure(1); 
for frame_index = 1:frame_index
    imshow(squeeze(imgs(frame_index, :, :, :)));
    title(sprintf('Time since stream start: %0.3f seconds (frame %d/%d)', ...
          imgs_t(frame_index) - imgs_t(1), frame_index, num_frames));
    drawnow;
end

fprintf('\n');


