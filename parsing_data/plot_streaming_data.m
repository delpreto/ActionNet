
%% Configuration
com_port = 'COM9';
baud_rate = 460800;

%% Connect to the glove
if exist('glove_serial', 'var')
    clear('glove_serial');
end
glove_serial = serialport(com_port, baud_rate);
flush(glove_serial);

%% Read data!
Fs_expected = 75; % Will be less than the glove's 100 Hz due to visualization overhead 
smoothing_duration_s = 0.1;
plot_duration_s = 20;
plot_update_period_s = 0.5;
N = plot_duration_s*Fs_expected;
strains = nan(N, 16);
% pressures = nan(N, 5);
accels = nan(N, 3);
% vrefs = nan(N, 1);
% currents = nan(N, 1);
times = nan(N, 1);
buffer_index = 1;

% figure(1); clf;
tiledlayout(3, 4);
flush(glove_serial);
count = 0;
t1 = now()*24*3600;
Fs_actual = Fs_expected;
while true
    % Read the latest line of data.
    data = char(readline(glove_serial));
    % Separate it into channels and convert the strings to doubles.
    data = strsplit(strrep(data, ' ', ''), ',');
    data = cellfun(@str2double, data);
    if length(data) ~= 19
        continue;
    end
    
    % Parse the columns into strains and accelerations.
    strains(buffer_index, :) = data(1:16);
%     pressures(n, :) = data(17:21);
    accels(buffer_index, :) = data(17:19);
%     vrefs(n, :) = data(25);
%     currents(n, :) = data(26);

    % Record the time since start.
    times(buffer_index, :) = now()*24*3600-t1;
    
    buffer_index = buffer_index+1;
    buffer_index = mod(buffer_index-1, N)+1;
    
    % Periodically update the plot.
    if mod(buffer_index, round(plot_update_period_s*Fs_actual)) == 0
        t = times - min(times);
        [t, I] = sort(t);
        s = strains(I, :);
%         p = pressures(I, :);
        a = accels(I, :);
        
        % Smooth and plot strains for the pinky finger.
        nexttile(1); 
        %plot(t([n:N, 1:n]), movmean(strains([n:N, 1:n], 1:3), 0.2*Fs_actual), '-');
        plot(t, movmean(s(:, 1:3), smoothing_duration_s*Fs_actual), '-', 'linewidth', 1.5);
        xlim([0 plot_duration_s]); 
%         ylim([0 200]);
        grid on;
        title('Pinky');
        
        % Smooth and plot strains for the ring finger.
        nexttile(2);
        plot(t, movmean(s(:, 4:6), smoothing_duration_s*Fs_actual), '-', 'linewidth', 1.5);
        xlim([0 plot_duration_s]);
        grid on;
        title('Ring');
        
        % Smooth and plot strains for the middle finger.
        nexttile(3);
        plot(t, movmean(s(:, 7:9), smoothing_duration_s*Fs_actual), '-', 'linewidth', 1.5);
        xlim([0 plot_duration_s]);
        grid on;
        title('Middle');
        
        % Smooth and plot strains for the index finger.
        nexttile(4); % index
        plot(t, movmean(s(:, 10:12), smoothing_duration_s*Fs_actual), '-', 'linewidth', 1.5);
        xlim([0 plot_duration_s]); 
        grid on;
        title('Index');
        
        % Smooth and plot strains for the thumb.
        nexttile(8); % thumb
        plot(t, movmean(s(:, 13:14), smoothing_duration_s*Fs_actual), '-', 'linewidth', 1.5);
        xlim([0 plot_duration_s]);
        grid on;
        title('Thumb');
        
        % Smooth and plot strains for the palm.
        nexttile(7); % palm
        plot(t, movmean(s(:, 15:16), smoothing_duration_s*Fs_actual), '-', 'linewidth', 1.5);
        xlim([0 plot_duration_s]);
        grid on;
        title('Palm');
        
        % Smooth and plot accelerations.
        nexttile(9, [1, 4]); % accel
        plot(t, movmean(a(:, :), smoothing_duration_s*Fs_actual), '-', 'linewidth', 1.5);
        xlim([0 plot_duration_s]);
        grid on;
        title('Accelerations'); 
        
        % Update the figure and flush the serial buffer.
        drawnow;
        flush(glove_serial);

        Fs_actual
    end
    % Update the sampling rate estimate.
    count = count+1;
    t2 = now()*24*3600;
    Fs_actual = count/(t2 - t1);
end












