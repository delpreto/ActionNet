%%%%%%%%%%%%
%
% Copyright (c) 2022 MIT CSAIL and Joseph DelPreto
%
% Permission is hereby granted, free of charge, to any person obtaining a copy
% of this software and associated documentation files (the "Software"), to deal
% in the Software without restriction, including without limitation the rights
% to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
% copies of the Software, and to permit persons to whom the Software is
% furnished to do so, subject to the following conditions:
%
% The above copyright notice and this permission notice shall be included in all
% copies or substantial portions of the Software.
%
% THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
% IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
% FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
% AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
% WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR
% IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
%
% See https://action-net.csail.mit.edu for more usage information.
% Created 2021-2022 for the MIT ActionNet project by Joseph DelPreto [https://josephdelpreto.com].
%
%%%%%%%%%%%%


%% EXPERIENCE
use_scatter = false;
headers = {'Overall', 'Cooking', 'Cutting', 'Washing', 'Organizing'};
subjects = {'S00', 'S01', 'S02', 'S03', 'S04', 'S05'};
data = [ % each row is a subject
    6	5	5	7	7
    8	8	6	8	8
    5	5	4	7	7
    5	8	6	8	6
    7	8	8	5	2
    3	3	3	8	8
    9	9	10	7	7
    7	5.5	4	9	6.5
    1	1	1	7	3
    9	8	7	10	10
    ];
figure(1); clf;
if use_scatter
    hold on;
    for i = 1:size(data, 2)
        h = plot(i*ones(size(data, 1), 1), data(:, i), '.', 'markersize', 50);
        dot_color = get(h, 'color');
        plot(i + 0.25*[-1 1], mean(data(:,i)) * [1 1], '-', 'linewidth', 3, 'color', clamp(dot_color*0.7, [0 1]));
    end
    xlim([0 i+1]);
    set(gca, 'XTickLabel', [{''} headers {''}]);
else
    boxchart(data);
    set(gca, 'XTickLabel', headers);
end
grid on;
box on;
title('Expertise Levels');
ylabel('Rating [0-10]');
ylim([0 10]);
set(gca, 'FontSize', 16);
update_figure_paper_size;

%% HANDEDNESS
data = {'Right', 'Right', 'Right', 'Right', 'Right', 'Right', 'Left', 'Right', 'Right', 'Right'};
num_left = sum(strcmpi(data, 'left'));
num_right = sum(strcmpi(data, 'right'));
fprintf('\n');
fprintf('\nRight handed: %d (%0.1f%%)', num_right, 100*num_right/(num_right+num_left));
fprintf('\nLeft  handed: %d (%0.1f%%)', num_left,  100*num_left/(num_right+num_left));

%% EYE DOMINANCE
data = {'Right', 'Right', 'Right', 'Left', 'Right', 'Left', 'Left', 'Right', 'Right', 'Right'};
num_left = sum(strcmpi(data, 'left'));
num_right = sum(strcmpi(data, 'right'));
fprintf('\n');
fprintf('\nRight eye: %d (%0.1f%%)', num_right, 100*num_right/(num_right+num_left));
fprintf('\nLeft  eye: %d (%0.1f%%)', num_left,  100*num_left/(num_right+num_left));


%% GENDER
data = {'M', 'M', 'F', 'M', 'F', 'M', 'M', 'M', 'M', 'F'};
num_male = sum(strcmpi(data, 'M'));
num_female = sum(strcmpi(data, 'F'));
fprintf('\n');
fprintf('\nMale  : %d (%0.1f%%)', num_male, 100*num_male/(num_male+num_female));
fprintf('\nFemale: %d (%0.1f%%)', num_female,  100*num_female/(num_male+num_female));

%% AGE
data = [
    31
    32
    31
    31
    25
    26
    26
    21
    24
    26
];
fprintf('\n');
fprintf('\nAge: %0.2f +- %0.2f (%d to %d)', mean(data), std(data), min(data), max(data));

%% SENSORS
use_scatter = false;
headers = {'Body-Trackers', 'Tactile', 'Gloves', 'Eye-Tracker', 'EMG'};
subjects = {'S00', 'S01', 'S02', 'S03', 'S04', 'S05'};
data = [ % each row is a subject
    1	7	7	3	0
    5	8	5	2.5	5
    1	8	3	1.5	7
    9	6	5	3	3
    8	8	7	1	2
    6	7.5	6.5	7.5	6
    7	nan	nan	2	6.5
    7	nan	nan	6	3
    4	nan	nan	0	3
    4	nan	nan	3	6
    ];
figure(2); clf;
if use_scatter
    hold on;
    for i = 1:size(data, 2)
        h = plot(i*ones(size(data, 1), 1), data(:, i), '.', 'markersize', 50);
        dot_color = get(h, 'color');
        plot(i + 0.25*[-1 1], mean(data(:,i)) * [1 1], '-', 'linewidth', 3, 'color', clamp(dot_color*0.7, [0 1]));
    end
    xlim([0 i+1]);
    set(gca, 'XTickLabel', [{''} headers {''}]);
else
    boxchart(data);
    set(gca, 'XTickLabel', headers);
end
grid on;
box on;
set(gca, 'XTickLabel', headers);
title('Sensor Obtrusiveness');
ylabel('Rating [0-10]');
ylim([0 10]);
set(gca, 'FontSize', 16);
update_figure_paper_size;

%% ROBOTS
use_scatter = false;

% headers = {'Prepare entire meal', 'Cut', 'Fetch', 'Load dishwasher', 'Unload dishwasher', 'Wash/dry plates', 'Put away tableware', 'Set/clear table', 'Cleaning'};
% subjects = {'S00', 'S01', 'S02', 'S03', 'S04', 'S05'};
% data = [ % each row is a subject
%         0.9	1.1	2.9	3.3	4.5	5.1	6.6	7.7	8.7	9.5	10.2	11.5	12.3	13.4	14.6	15.6	16.2	17.4
%         0.4	1.8	2.2	3.2	4.6	5.3	6.5	7.2	8.3	10.0	10.2	11.2	12.3	14.0	14.8	16.0	16.2	17.6
%         0.3	1.6	2.2	3.1	4.8	5.8	6.7	7.9	9.0	9.1	10.9	11.0	12.6	13.8	14.3	15.4	16.4	17.0
%         0.4	1.2	2.4	3.4	4.5	5.7	6.2	7.8	8.3	9.2	10.3	11.1	12.8	13.7	14.2	15.7	16.8	17.6
%         0.5	1.6	2.9	3.9	4.3	5.5	6.1	7.7	8.8	9.4	10.1	11.7	12.7	13.6	14.8	15.5	17.0	17.1
%         0.2	1.9	2.1	3.7	4.3	5.6	6.7	7.2	8.2	9.7	10.2	11.2	12.9	13.3	14.7	15.6	16.6	17.5
%     ];
% figure(3); clf;
% boxplotGroup({data(:,1:9), data(:,10:18)}, 'primaryLabels', {'Coll.', 'Auto'}, 'secondaryLabels', headers);
% grid on;
% % set(gca, 'XTickLabel', headers);
% title('Robot Assistant Desirability');
% ylabel('Rating [0-10]');
ylim([0 10]);

headers = {'Meal', 'Cut', 'Fetch', 'Load', 'Unload', 'Wash', 'Return', 'Table', 'Clean'};
subjects = {'S00', 'S01', 'S02', 'S03', 'S04', 'S05'};
data = [ % each row is a subject
        10	10	10	9	7	8	9	6	10	6	8	6	9	6	7	8	10	9
        10	10	10	0	5	5	10	10	10	10	0	0	10	10	10	10	10	10
        7	8	10	1	6	6.75	2.75	10	3	10	2.75	10	1	10	1	10	8	8.75
        0	1	5	3	2	2	8	0	8	0	1	1	4	4	4	4	7	8
        10	2	10	2	10	2	10	2	10	2	2	2	2	8	10	3	10	10
        10	10	10	10	10	10	10	10	10	10	10	10	10	10	10	10	10	10
        7	9	4	4	7	9	10	10	10	10	10	10	5	10	10	10	10	10
        2	8	8	nan	3	nan	10	9	10	9	10	9	10	10	10	10	10	10
        9	7	9	7	9	7	10	10	10	10	7	10	8	7	4	6	10	10
        10	8	10	10	10	10	10	10	10	10	10	8	10	10	10	8	10	10
    ];
figure(3); clf;
if use_scatter
    hold on;
    x = 1;
    for i = 1:2:size(data, 2)
        h = plot(x*ones(size(data, 1), 1), data(:, i), '.', 'markersize', 50);
        dot_color = get(h, 'color');
        plot(x + 0.25*[-1 1], mean(data(:,i)) * [1 1], '-', 'linewidth', 3, 'color', clamp(dot_color*0.7, [0 1]));
        x = x + 1
    end
    x = x+1;
    for i = 2:2:size(data, 2)
        h = plot(x*ones(size(data, 1), 1), data(:, i), '.', 'markersize', 50);
        dot_color = get(h, 'color');
        plot(x + 0.25*[-1 1], mean(data(:,i)) * [1 1], '-', 'linewidth', 3, 'color', clamp(dot_color*0.7, [0 1]));
        x = x + 1
    end
    xlim([0 x]);
    set(gca, 'XTickLabel', [{''} headers {''} headers {''}]);
else
    boxplotGroup({data(:,1:2), data(:,3:4), data(:,5:6), data(:,7:8), data(:,9:10), data(:,11:12), data(:,13:14), data(:,15:16), data(:,17:18)}, ...
                 'primaryLabels', headers, 'secondaryLabels', {'Autonomously', 'Collaboratively'});
end
grid on;
box on;
title('Robot Assistant Desirability');
ylabel('Rating [0-10]');
ylim([0 10]);
set(gca, 'FontSize', 10);
update_figure_paper_size;

%% WORKLOAD
data = [ % each row is a subject
        5	3	2	1	3	1
        0	0	0	0	0	0
        2	6	4	2.5	6.5	0.5
        4	3	1	6	4	1
        3	7	0	0	4	1
        7	7	5	1.5	7	1.5
        5	7.5	6.5	1	4	5
        7	6	5	2	7	6
        6	4	3	2	5	7
        2	2	4	1.5	4	2
    ];
data = mean(data, 2);
fprintf('\n');
fprintf('\nWorkload: %0.2f +- %0.2f (%0.0f to %0.0f)', mean(data), std(data), min(data), max(data));

%%
fprintf('\n');




