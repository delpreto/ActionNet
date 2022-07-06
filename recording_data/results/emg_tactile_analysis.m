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


corr_coeffs = [];
activity = 'slice_potatoes'; % slice_potatoes slice_cucumbers pour_water

for subject_index = 1:5

switch subject_index
    case 1
    filepath = 'P:\MIT\Lab\Wearativity\data\experiments\2022-06-07_experiment_S00\2022-06-07_18-10-55_actionNet-wearables_S00\2022-06-07_18-11-37_streamLog_actionNet-wearables_S00.hdf5';
    switch activity
        case 'slice_potatoes'
            start_time_s = 1.654640902386591E9;
            end_time_s = 1.6546410203133373E9;
        case 'slice_cucumbers'
            start_time_s = 1654640434.59;
            end_time_s = 1654640590.17;
        case 'pour_water'
            start_time_s = 1654641882.59;
            end_time_s = 1654641963.93;
    end
    
    case 2
        if strcmpi(activity, 'pour_water')
            filepath = 'P:\MIT\Lab\Wearativity\data\experiments\2022-06-13_experiment_S02\2022-06-13_22-34-45_actionNet-wearables_S02\2022-06-13_22-35-11_streamLog_actionNet-wearables_S02.hdf5';
        else
            filepath = 'P:\MIT\Lab\Wearativity\data\experiments\2022-06-13_experiment_S02\2022-06-13_21-47-57_actionNet-wearables_S02\2022-06-13_21-48-24_streamLog_actionNet-wearables_S02.hdf5';
        end
    switch activity
        case 'slice_potatoes'
            start_time_s = 1.6551731273054476E9;
            end_time_s = 1.6551731936853428E9;
        case 'slice_cucumbers'
            start_time_s = 1655172507.67;
            end_time_s = 1655172650.98;
        case 'pour_water'
            start_time_s = 1655174387.57;
            end_time_s = 1655174432.44;
    end
    
    case 3
        if strcmpi(activity, 'pour_water')
            filepath = 'P:\MIT\Lab\Wearativity\data\experiments\2022-06-14_experiment_S03\2022-06-14_13-52-21_actionNet-wearables_S03\2022-06-14_13-52-57_streamLog_actionNet-wearables_S03.hdf5';
        else
            filepath = 'P:\MIT\Lab\Wearativity\data\experiments\2022-06-14_experiment_S03\2022-06-14_13-11-44_actionNet-wearables_S03\2022-06-14_13-12-07_streamLog_actionNet-wearables_S03.hdf5';
        end
    switch activity
        case 'slice_potatoes'
            start_time_s = 1.6552284775788276E9;
            end_time_s = 1.6552286755249252E9;
        case 'slice_cucumbers'
            start_time_s = 1655227654.60;
            end_time_s = 1655228172.90;
        case 'pour_water'
            start_time_s = 1655229838.48;
            end_time_s = 1655229884.40;
    end
    
    case 4
    filepath = 'P:\MIT\Lab\Wearativity\data\experiments\2022-06-14_experiment_S04\2022-06-14_16-38-18_actionNet-wearables_S04\2022-06-14_16-38-43_streamLog_actionNet-wearables_S04.hdf5';
    switch activity
        case 'slice_potatoes'
            start_time_s = 1.6552408396758137E9;
            end_time_s = 1.655240926072642E9;
        case 'slice_cucumbers'
            start_time_s = 1655240291.10;
            end_time_s = 1655240453.91;
        case 'pour_water'
            start_time_s = 1655241531.79;
            end_time_s = 1655241571.09;
    end
    
    case 5
    filepath = 'P:\MIT\Lab\Wearativity\data\experiments\2022-06-14_experiment_S05\2022-06-14_20-45-43_actionNet-wearables_S05\2022-06-14_20-46-12_streamLog_actionNet-wearables_S05.hdf5';
    switch activity
        case 'slice_potatoes'
            start_time_s = 1.6552551876786969E9;
            end_time_s = 1.6552553283361132E9;
        case 'slice_cucumbers'
            start_time_s = 1655254833.34;
            end_time_s = 1655254895.23;
        case 'pour_water'
            start_time_s = 1655255980.56;
            end_time_s = 1655256018.71;
    end

end

emg_data = double(h5read(filepath, '/myo-right/emg/data')');
emg_time_s = h5read(filepath, '/myo-right/emg/time_s')';
i = emg_time_s >= start_time_s & emg_time_s <= end_time_s;
emg_data = emg_data(i,:);
emg_time_s = emg_time_s(i, :);
emg_Fs = (size(emg_data, 1)-1)/(max(emg_time_s) - min(emg_time_s));
% emg_data_sum = sum(abs(emg_data), 2);
emg_data_sum = sum(abs(emg_data(:, :)), 2);
emg_data_sum = emg_data_sum - min(emg_data_sum);
emg_data_sum = emg_data_sum/max(emg_data_sum);
emg_data_envelope = emg_data_sum;
% emg_data_envelope = lowpass(abs(emg_data_sum), 0.00001, emg_Fs);
[b,a] = butter(5, 0.5/(emg_Fs/2), 'low');
emg_data_envelope = filter(b, a, emg_data_sum);
% emg_data_envelope = movmean(emg_data_envelope, [round(10*emg_Fs), 0]);
emg_data_envelope = emg_data_envelope - min(emg_data_envelope);
emg_data_envelope = emg_data_envelope/max(emg_data_envelope);
% emg_data_envelope = movmean(abs(emg_data_sum), [round(0.1*emg_Fs), 0]);
% emg_data_envelope = emg_data_envelope/max(emg_data_envelope);

% figure(1); clf;
% plot(emg_time_s-min(emg_time_s), emg_data_sum);
% hold on;
% plot(emg_time_s-min(emg_time_s), emg_data_envelope);

tactile_data = permute(double(h5read(filepath, '/tactile-glove-right/tactile_data/data')), [3 1 2]);
tactile_data(tactile_data >= 1024) = 0;
tactile_time_s = double(h5read(filepath, '/tactile-glove-right/tactile_data/time_s')');
i = tactile_time_s >= start_time_s & tactile_time_s <= end_time_s;
tactile_data = tactile_data(i,:);
tactile_time_s = tactile_time_s(i, :);
tactile_Fs = (size(tactile_data, 1)-1)/(max(tactile_time_s) - min(tactile_time_s));
tactile_data_sum = squeeze(sum(sum(tactile_data, 2), 3));
tactile_data_envelope = tactile_data_sum;
% [b,a] = butter(5, 2/(tactile_Fs/2), 'low');
% tactile_data_envelope = filter(b, a, tactile_data_sum);
tactile_data_envelope = movmean(tactile_data_envelope, [round(1*tactile_Fs), 0]);
tactile_data_envelope = tactile_data_envelope - min(tactile_data_envelope);
tactile_data_envelope = tactile_data_envelope/max(tactile_data_envelope);

% figure(1); clf;
% % h1 = subplot(2,1,1);
% plot(emg_time_s, emg_data_envelope);
% grid on; hold on;
% % h2 = subplot(2,1,2);
% plot(tactile_time_s, tactile_data_envelope);
% grid on;
% % linkaxes([h1 h2], 'x');
% % xlim([500 560]+min(emg_time_s));


time_s = tactile_time_s;
time_s = time_s(time_s >= min(emg_time_s) & time_s <= max(emg_time_s));
emg_data_envelope_ts = timeseries(emg_data_envelope, emg_time_s);
tactile_data_envelope_ts = timeseries(tactile_data_envelope, tactile_time_s);
emg_data_envelope_ts_resampled = resample(emg_data_envelope_ts, time_s);
tactile_data_envelope_ts_resampled = resample(tactile_data_envelope_ts, time_s);

time_s = time_s - min(time_s);
emg_data_envelope_ts_resampled.Time = time_s;
tactile_data_envelope_ts_resampled.Time = time_s;

if subject_index == 1
figure(2); clf;
set(gcf, 'position', [1 49 1152 604.333333333333]);
end
subplot(3,2,subject_index);
plot(emg_data_envelope_ts_resampled, 'linewidth', 1.5);
grid on; hold on;
% h2 = subplot(2,1,2);
plot(tactile_data_envelope_ts_resampled, '-', 'linewidth', 1.5, 'color', 'k');
grid on;

set(gca, 'FontSize', 14);
if subject_index == 1
legend({'Overall Muscle Activation', 'Overall Tactile Force Reading'});
end
if subject_index == 3
    ylabel('Normalized Readings');
else
    ylabel('');
end
if any(subject_index == [4 5])
    xlabel('Time [s]');
else
    xlabel('');
end
title('');
switch activity
    case 'slice_potatoes'
        sgtitle('Muscle Activation and Tactile Feedback: Slicing Potatoes');
    case 'slice_cucumbers'
        sgtitle('Muscle Activation and Tactile Feedback: Slicing Cucumbers');
    case 'pour_water'
        sgtitle('Muscle Activation and Tactile Feedback: Pouring Water');
end
xlim([min(time_s) max(time_s)]);
% if subject_index == 5
% tightfig;
% end

% figure(3); clf;
% plot(emg_data_envelope_ts_resampled.Data, tactile_data_envelope_ts_resampled.Data, '.');
% ylabel('EMG');
% xlabel('Tactile');

[R, P] = corrcoef(emg_data_envelope_ts_resampled.Data, tactile_data_envelope_ts_resampled.Data);
[r, lags] = xcorr(emg_data_envelope_ts_resampled.Data, tactile_data_envelope_ts_resampled.Data, 'biased');
% figure(4); clf;
% plot(lags, r);
corr_coeffs(end+1) = R(1,2);




end

corr_coeffs
mean(corr_coeffs)
std(corr_coeffs)

update_figure_paper_size;

