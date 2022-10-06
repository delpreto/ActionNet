
bag_filename = 'kitchen_depth-depth_2022-06-07-17-31-35.bag';
topic_data = '/kitchen_depth/depth/color/points';
topic_extrinsics = '/kitchen_depth/extrinsics/depth_to_color';
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
bag = select(bag, 'Time', [start_time_s, end_time_s], 'Topic', {topic_data, topic_extrinsics});

% Separate the data from the extrinsics.
extrinsics_bag = select(bag, 'Topic', topic_extrinsics);
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
fprintf('Creating point cloud structs from the messages... ');
point_clouds_struct = readMessages(bag, 'DataFormat', 'struct');
fprintf('done\n');

%% Convert to points with XYZ coordinates and RGB colors.
fprintf('Extracting XYZ coordinates and RGB colors... ');
point_clouds_xyz = cell(num_frames, 1);
point_clouds_rgb = cell(num_frames, 1);
point_clouds_t = nan(num_frames, 1);
for frame_index = 1:num_frames
    if mod(frame_index, round(length(point_clouds_struct)/100)) == 0
        fprintf('\n  Processed %05d/%d frames (%0.1f%%) ', (frame_index-1), num_frames, 100*(frame_index-1)/num_frames);
    end
    point_clouds_xyz{frame_index} = rosReadXYZ(point_clouds_struct{frame_index}, 'PreserveStructureOnRead', false);
    point_clouds_rgb{frame_index} = rosReadRGB(point_clouds_struct{frame_index}, 'PreserveStructureOnRead', false);
    % Note that the timestamps are stored as integers, so need to convert
    %  to doubles before dividing to convert nanoseconds to seconds.
    t_s = double(point_clouds_struct{frame_index}.Header.Stamp.Sec);
    t_ns = double(point_clouds_struct{frame_index}.Header.Stamp.Nsec);
    point_clouds_t(frame_index) = t_s + t_ns/1e9;
end
fprintf('done\n');

%% Plot the points
figure(1); 
for frame_index = 1:num_frames
    cla;
    % Note that xyz dimensions are swapped to allow for better viewing angles. 
    h = scatter3(point_clouds_xyz{frame_index}(:,3), ...
                 point_clouds_xyz{frame_index}(:,1), ...
                 -point_clouds_xyz{frame_index}(:,2), ...
                 2, point_clouds_rgb{frame_index});
    xlabel('z'); ylabel('x'); zlabel('y');
    ylim([-1.2, 1.2]);
    zlim([-0.2, 0.5]);
    xlim([0, 2]);
    view(-65, 10);
    axis square
    title(sprintf('Time since stream start: %0.3f seconds (frame %d/%d)', ...
          point_clouds_t(frame_index) - point_clouds_t(1), frame_index, num_frames));
    drawnow;
end

fprintf('\n');


