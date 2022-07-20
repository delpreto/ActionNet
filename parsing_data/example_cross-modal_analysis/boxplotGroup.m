function handles = boxplotGroup(varargin)
% BOXPLOTGROUP groups boxplots together with horizontal space between groups.
%   boxplotGroup(x) receives a 1xm cell array where each element is a matrix with
%   n columns and produces n groups of boxplot boxes with m boxes per group.
%
%   boxplotGroup(ax,x,___) specifies the axis handle, otherwise current axis is used.
%
%   boxplotGroup(___,'interGroupSpace',d) separates groups by d units along the x axis
%   where d is a positive, scalar integer (default = 1)
%
%   boxplotGroup(___,'groupLines', true) adds vertical divider lines between groups
%   (requires >=r2018b).
%
%   boxplotGroup(___,'primaryLabels', c) specifies the x tick label for each boxplot.
%   c is a string array or cell array of characters and must have one element per
%   box or one element per group-member. When undefined or when c is an empty cell {},
%   the x-axis is labeled with default x-tick labels.
%
%   boxplotGroup(___,'secondaryLabels', s) specifies the group labels for the boxplot
%   groups.  s is a string array or cell array of characters and must have one element
%   per group (see 'groupLabelType'). Ignored when s is an empty cell {}.
%
%   boxplotGroup(___,'groupLabelType', str) specifies how to label the groups by one of
%   the following options.
%    * 'horizontal': Group labels will be centered under the primary labels using a 2nd
%       invisible axis underlying the main axis (not supported in uifigures). To remove
%       the primary labels and only show secondary labels, set primary labels to empty
%       cell-strings (e.g. {'','',''}) or strings without characters (e.g. ["" "" ""]).
%    * 'vertical': Group labels will be vertical, between groups (requires Matlab >=2018b)
%    * 'both': Both methods will be used.
%
%   boxplotGroup(___, 'PARAM1', val1, 'PARAM2, val2, ...) sends optional name/value pairs
%   to the boxplot() function. Accepted parameters are BoxStyle, Colors, MedianStyle,
%   Notch, OutlierSize, PlotStyle, Symbol, Widths, DataLim, ExtremeMode, Jitter, and Whisker.
%   See boxplot documentation for details.
%
%   boxplotGroup(___, 'Colors', ___, 'GroupType', type) determines how to apply
%   colors to the groups.  'Colors' is a property of boxplots (see boxplot documentation).
%   When the colors value specifies multiple colors, the 'GroupType' determines how
%   the colors are distributed based on the following two options.
%    * 'betweenGroups' assigns color n to the n^th boxplot within each group (default).
%    * 'withinGroups' assigns color n to all boxplots within the n^th group.
%
%   h = boxplotGroup(___) outputs a structure of graphics handles.
%
% NOTE: If you're working with a grouping variable 'g', use the syntax boxplot(x,g) along
%   with the "Group Appearance" options described in Matlab's boxplot() documentation.
%   https://www.mathworks.com/help/stats/boxplot.html#d118e146984
%
% EXAMPLES:
% data = {rand(100,4), rand(20,4)*.8, rand(1000,4)*1.2};
%
% Required inputs
%   boxplotGroup(data)
%
% Set space between groups
%   boxplotGroup(data, 'interGroupSpace', 3)
%
% Specify labels and draw divider line
%   boxplotGroup(data, 'groupLines', true, 'PrimaryLabels', {'a' 'b' 'c'},...
%       'SecondaryLabels', {'Lancaster', 'Cincinnati', 'Sofia', 'Rochester'})
%
% Label groups with vertical lables
%   boxplotGroup(data, 'PrimaryLabels', {'a' 'b' 'c'}, 'SecondaryLabels', ...
%       {'Lancaster', 'Cincinnati', 'Sofia', 'Rochester'}, 'groupLabelType', 'vertical')
%
% Pass additional boxplot properties
%   boxplotGroup(data, 'PrimaryLabels', {'a' 'b' 'c'}, 'SecondaryLabels', ...
%       {'Lancaster', 'Cincinnati', 'Sofia', 'Rochester'}, 'groupLabelType', 'vertical', ...
%       'BoxStyle', 'filled', 'PlotStyle', 'Compact')
%
%
% Contact adam.danz@gmail.com for questions, bugs, suggestions, and high-fives.
% Copyright (c) 2020, Adam Danz  adam.danz@gmail.com
% All rights reserved
% Source: https://www.mathworks.com/matlabcentral/fileexchange/74437-boxplotgroup

% Changes history
% 200306 - v1.0.0 first release.
% 200308 - v1.1.0 Added recommendation to use boxplot() with grouping variable.
%                 Added axis handle as input to boxplot() call. Linkaxes changed
%                 from x to xy. Set axis2.Units to axis.Units.  Using linkprop
%                 to link position etc of main axis and axis2. Added DeleteFcn
%                 to main axis. Disabled toolbar for axis2. Added listener to
%                 resize axis2 when main axis is resized. Changes to help section.
% 200309 - v1.2.0 When 2nd axis is added, main axis is set to current axis.
% 200309 - v1.2.1 Suppress linkprops() and changes to toolbar suppression to work
%                 with versions prior to r2018b.
% 200309 - v1.2.2 Instead of creating new axis, default axis is gca().
% 210427 - v2.0.0 oncleanup returns hold state instead of conditional.  Added GroupType
%                 option and colorexpansion. Suppresses output unless requested.
%                 Checks matlab vs with xline(). Removed listener, storing hlink in axis.
%                 boxplot name-val arg check. Removing boxplot placeholders. XTicks now auto
%                 if labels aren't provided. Outputs now include boxplotGroup; vertical
%                 labels now the same fontsize and weight as axis font; Primary and secondary
%                 labels can be empty cell to ignore. Secondary labels now match ax1 font size,
%                 weight and name.

%% Check for axis handle in first input
if ~isempty(varargin) && ~isempty(varargin{1}) && isobject(varargin{1}(1)) % [3]
    if isgraphics(varargin{1}(1), 'axes')
        % first input is an axis
        h.axis = varargin{1} ;
        varargin(1) = [];
    else
        error('MATLAB:hg:InvalidHandle', 'Invalid handle')
    end
else
    h.axis = [];
end

%% Parse inputs
p = inputParser();
p.FunctionName = mfilename;
p.KeepUnmatched = true;	%accept additional parameter value inputs (passed to boxplot())
addRequired(p, 'x', @(x)validateattributes(x,{'cell'},{'row','nonempty'}))
addParameter(p, 'interGroupSpace', 1, @(x)validateattributes(x,{'double'},{'scalar','integer'}))
addParameter(p, 'primarylabels', [], @(x)validateattributes(x,{'string','cell'},{}))
addParameter(p, 'secondarylabels', [], @(x)validateattributes(x,{'string','cell'},{}))
addParameter(p, 'groupLines', false, @(x)validateattributes(x,{'logical','double'},{'binary'}))
addParameter(p, 'groupLabelType', 'Horizontal', @(x)ischar(validatestring(lower(x),{'vertical','horizontal','both'})))
addParameter(p, 'GroupType', 'betweenGroups', @(x)ischar(validatestring(lower(x),{'betweengroups','withingroups'})))
parse(p,varargin{:})

% Prepare the unmatched boxplot() parameters.
% If a param is passed that isn't accepted by boxplot(), an error is thrown from boxplot() function.
unmatchNameVal = reshape([fieldnames(p.Unmatched)'; struct2cell(p.Unmatched)'], 1, []);

% Check boxplot name-value parameters; group params, Position, and labels are not accepted.
supportedParams = {'BoxStyle','Colors','MedianStyle','Notch','OutlierSize','PlotStyle','Symbol','Widths', ...
    'DataLim','ExtremeMode','Jitter','Whisker'};
argOK = arrayfun(@(i)any(strncmpi(unmatchNameVal{i},supportedParams,numel(unmatchNameVal{i}))),...
    1:2:numel(unmatchNameVal)); % look for partial match
assert(all(argOK),'Parameter(s) not accepted in %s: [%s].', ...
    mfilename, strjoin(unmatchNameVal(find(~argOK)*2-1),', '))

% Check that each element of x is a matrix
assert(all(cellfun(@ismatrix, p.Results.x)), 'All elements of the cell array ''x'' must be a matrix.')
% Check that each matrix contains the same number of columns.
assert(numel(unique(cellfun(@(m)size(m,2),p.Results.x))) == 1, ...
    ['All elements of the cell array ''x'' must contain the same number of columns. '...
    'Pad the matricies that contain fewer columns with NaN values.']);

nargoutchk(0,1)

%% Compute horizontal spacing & check labels
nGroups = size(p.Results.x{1},2);       % number of columns of data / number of groups
nMembers = numel(p.Results.x);          % number of members per group
maxX = ((nMembers + p.Results.interGroupSpace) * nGroups) - p.Results.interGroupSpace;
xInterval = nMembers + p.Results.interGroupSpace;

% Check that labels (if any) are the right size
% PrimaryLabels: either 1 per group-member or 1 for each bar
if ~isempty(p.Results.primarylabels)
    assert(ismember(numel(p.Results.primarylabels),[nMembers, nMembers*nGroups]), ...
        sprintf(['The number of primary labels must equal either the number of bars per group (%d) '...
        'or the number of total bars (%d).'], nMembers, nMembers*nGroups))
end
% SecondaryLabels: 1 per group
if ~isempty(p.Results.secondarylabels)
    assert(isequal(numel(p.Results.secondarylabels),nGroups), ...
        sprintf('The number of secondary labels must equal either the number groups (%d).',nGroups))
end

% If all primary labels are empty chars do not add the newline to secondary labels.
if ~isempty(p.Results.primarylabels) &&  all(cellfun(@isempty,cellstr(p.Results.primarylabels)))
    horizSecondaryLabelAddon = '';
else
    horizSecondaryLabelAddon = '\newline';
end

%% Set colors
% Assumes ColorGroup property is not specified.
colorsIdx = strcmpi('Colors',unmatchNameVal);
if any(colorsIdx)
    cvalIdx = find(colorsIdx,1,'first')+1;
    if isempty(unmatchNameVal{cvalIdx})
        % Colors val is empty; remove Colors name-val pair
        unmatchNameVal(cvalIdx-[1,0]) = [];
    else
        unmatchNameVal{cvalIdx} = colorexpansion(unmatchNameVal{cvalIdx}, p, nGroups, nMembers);
    end
end

%% Do plotting
if isempty(h.axis)
    h.axis = gca();
end
h.figure = ancestor(h.axis,'figure');
isTiledLayout = strcmpi(h.axis.Parent.Type,'tiledlayout');
if isTiledLayout % [6]
    origTLOState = warning('query', 'MATLAB:handle_graphics:Layout:NoPositionSetInTiledChartLayout');
    TLOcleanup = onCleanup(@()warning(origTLOState));
    warning('off','MATLAB:handle_graphics:Layout:NoPositionSetInTiledChartLayout')
end

% Store pre-existing boxplot object handles
bptag = 'boxplot'; % tag Matlab assigns to bp group
bpobjPre = findobj(h.axis,'tag',bptag);

originalHoldStatus = ishold(h.axis);
holdStates = {'off','on'};
returnHoldState = onCleanup(@()hold(h.axis,holdStates{originalHoldStatus+1}));
hold(h.axis, 'on')

x = cell(1,nMembers);
existingTextObjs = findobj(h.axis,'Type','Text');
for i = 1:nMembers
    x{i} = i : xInterval : maxX;
    temp = nan(size(p.Results.x{i},1), max(x{i}));
    temp(:,x{i}) = p.Results.x{i};
    boxplot(h.axis, temp, unmatchNameVal{:})
end

% Remove dummy boxplots placeholders
bpobjNew = findobj(h.axis,'tag',bptag);
bpobjNew(ismember(bpobjNew, bpobjPre)) = [];
for g = 1:numel(bpobjNew)
    tags = unique(get(bpobjNew(g).Children,'tag'),'stable');
    tags(cellfun(@isempty,tags)) = [];
    for j = 1:numel(tags)
        obj = findobj(bpobjNew(g),'tag',tags{j});
        obj(~isprop(obj,'YData')) = [];
        YData = get(obj,'YData');
        if ~iscell(YData)
            YData = {YData};
        end
        isDummy = cellfun(@(c)all(isnan(c),2),YData);
        delete(obj(isDummy))
    end
end

axis(h.axis, 'tight')
limGap = (p.Results.interGroupSpace+1)/2;
set(h.axis,'XTickMode','Auto','XTickLabelMode','Auto','xlim',[1-limGap, maxX+limGap]) %[1]
yl = ylim(h.axis);
ylim(h.axis, yl + [-range(yl)*.05, range(yl)*.05])

% Remove boxplot's text-tics [1]
allTextObjs = findobj(h.axis,'Type','Text');
isBoxplotText = ~ismember(allTextObjs,existingTextObjs);
set(allTextObjs(isBoxplotText), 'String','','Visible','off')

% Set primary labels if provided
if ~isempty(p.Results.primarylabels)
    h.axis.XTick = sort([x{:}]);
    h.axis.XTickLabel = p.Results.primarylabels;
end
% Set secondary labels if provided
vertLinesDrawn = false;
groupLabelType = p.Results.groupLabelType;
if ~isempty(p.Results.secondarylabels)
    if any(strcmpi(groupLabelType, {'horizontal','both'}))
        % Try to detect figure type [4]
        if verLessThan('Matlab','9.0')      %version < 16a (release of uifigs)
            isuifig = @(~)false;
        elseif verLessThan('Matlab','9.5')  % 16a <= version < 18b
            isuifig = @(h)~isempty(matlab.ui.internal.dialog.DialogHelper.getFigureID(h));
        else                                % version >= 18b (written in r21a)
            isuifig = @(h)matlab.ui.internal.isUIFigure(h) && ~isprop(h,'LiveEditorRunTimeFigure');
        end
        isUIFigure = isuifig(h.figure);
        if isUIFigure
            groupLabelType = 'vertical';
            warning('BOXPLOTGRP:uifig','''Horizontal'' GroupLabelType is not supported with UIFIgures. GroupLabelType was changed to ''vertical''.')
        else
            % Tick label rotation must be 0 if using both primary & secondary horizontal labels
            h.axis.XAxis.TickLabelRotation = 0;
            % Compute x position of secondary labels
            if isa(h.axis,'matlab.ui.control.UIAxes')
                axFcn = @uiaxes;
            else
                axFcn = @axes;
            end
            if verLessThan('Matlab','9.8') %r2020a
                posProp = 'Position';
            else
                posProp = 'InnerPosition';
            end
            secondaryX = (nMembers : nMembers + p.Results.interGroupSpace : maxX) - (nMembers-1)/2;
            secondaryLabels = strcat(horizSecondaryLabelAddon,p.Results.secondarylabels); %[2]
            h.axis2 = axFcn(h.figure,'Units',h.axis.Units, 'OuterPosition', h.axis.OuterPosition, ...
                'ActivePositionProperty', h.axis.ActivePositionProperty,'xlim', h.axis.XLim, ...
                'TickLength', [0 0], 'ytick', [], 'Color', 'none', 'XTick', secondaryX, ...
                'TickLabelInterpreter','tex','XTickLabel', secondaryLabels,'HitTest','off',...
                'XTickLabelRotation',0,'box','off','FontSize',h.axis.FontSize,...
                'FontWeight',h.axis.FontWeight,'FontName',h.axis.FontName);
            h.axis.(posProp)([2,4]) = h.axis2.(posProp)([2,4]); % make room in original axes for 2ndary labels.
            h.axis2.(posProp)([1,3]) = h.axis.(posProp)([1,3]); % let original axis control lateral placement
            h.axis2.UserData.hlink = linkprop([h.axis, h.axis2],...
                {'Units',posProp,'ActivePositionProperty','Parent'}); % [5]
            linkaxes([h.axis, h.axis2], 'xy')
            if ~isUIFigure % [4]
                uistack(h.axis2, 'down')
            end
            if isprop(h.axis2, 'Toolbar')
                h.axis2.Toolbar.Visible = 'off'; % ver >= r2018b
            end
            h.axis2.XRuler.Axle.Visible = 'off';
            h.axis2.YRuler.Axle.Visible = 'off';
            h.axis.DeleteFcn = @(~,~)delete(h.axis2); % Delete axis2 if main axis is deleted
            set(h.figure,'CurrentAxes',h.axis)
        end
    end
    if any(strcmpi(groupLabelType, {'vertical','both'})) && ~verLessThan('Matlab','9.5') % r18b
        spaces = setdiff(1-p.Results.interGroupSpace : maxX, [x{:}]);
        endSpaceIdx = [diff(spaces),2] > 1;
        midSpace = spaces(endSpaceIdx) - (p.Results.interGroupSpace-1)/2;
        h.xline = arrayfun(@(x)xline(h.axis, x,'FontSize',h.axis.FontSize,...
            'FontWeight',h.axis.FontWeight,'FontName',h.axis.FontName),midSpace);
        set(h.xline(:), {'Label'}, cellstr(p.Results.secondarylabels(:))) % cellstr in case lbls are str
        vertLinesDrawn = true;
    end
end

% Draw vertical lines if requested and if they don't already exist.
if p.Results.groupLines && ~vertLinesDrawn && ~verLessThan('Matlab','9.5') %r18b
    spaces = setdiff(1:maxX+p.Results.interGroupSpace, [x{:}]);
    endSpaceIdx = [diff(spaces),2] > 1;
    midSpace = spaces(endSpaceIdx) - (p.Results.interGroupSpace-1)/2;
    h.xline = arrayfun(@(x)xline(h.axis, x,'-k'),midSpace);
end

clear('returnHoldState','TLOcleanup')

%% Return output only if requested
if nargout>0
    % Get and organize new boxplot groups
    bpobjPost = findobj(h.axis,'tag',bptag);
    h.boxplotGroup = bpobjPost(~ismember(bpobjPost, bpobjPre));
    handles = h;
end

function c = colorexpansion(colors, p, nGroups, nMembers)
% colors is color data. As of r2021a, boxplot 'Colors' can be RGB triplet/matrix
%   char vec, or string scalar of chars ("rgb"). Long color names is not accepted
%   by boxplot. 'colors' cannot be empty for this function.
% c: if 'colors' specifies more than 1 color, c is the color scheme expanded according
%   to GroupType. Otherwise, c is the same as colors.
% Other inputs defined in main func.
if isnumeric(colors) && size(colors,1)>1
    basecolors = colors;
    
elseif (ischar(colors) || isa(colors,'string')) && numel(char(colors))>1
    basecolors = char(colors);
    basecolors = basecolors(:); % col vec
    
else
    % If colors is not numeric, char, or string let boxplot throw the error.
    % If colors specifies only 1 color, copy colors to output.
    c = colors;
    return
end
isBetweenGroups = strcmpi('betweenGroups', p.Results.GroupType);
n = size(basecolors,1);
getRowIdx = @(n,m)round(mod(1:n,m+1E-08));
if isBetweenGroups
    % The first nMembers of colors will be used
    % Let boxplot do the expansion.
    rowNum = getRowIdx(nMembers,n);
    c = [basecolors(rowNum,:);repmat(basecolors(1,:),p.Results.interGroupSpace,1)];
else
    % The first nGroups colors will be used
    rowNum = getRowIdx(nGroups,n);
    c = repelem(basecolors(rowNum,:),nMembers+p.Results.interGroupSpace,1);
end
if ischar(c)
    c = c';
end


%% Footnotes
% [1] Matlab's boxplot labels are text objects (why????). Deleting them isn't a good options
%   since several listeners are built-in to respond to changes to the plot.  Documentation
%   suggests setting xticklabel to remove them. When this is done, the text ticks visibilty
%   are turned off by a listener and the invisible text objects remain.  findobj(h.axis,'type','text')
%   However, in older releases (ie, r16a), setting xticklabel does not remove the text objs so
%   their strings are replaced with empties and visibility to off.
% [2] The carriage return is not processed in uifigures (as of r2021a); same with alternative below.
%   Alternative: cellfun(@(str){sprintf('\\newline%s',str)},p.Results.secondarylabels);
% [3] These conditions will catch a deleted axes.  See https://www.mathworks.com/matlabcentral/answers/254118#answer_680067
% [4] uistack not supported in uifigs (as of r21a). Also, the multi-line x-ticks are not interpreted
%   correctly with uifigures.  There isn't a good way to detect uifigs across
%   all releases. See https://www.mathworks.com/matlabcentral/answers/348387
% [5] setappdata had much slower response when dragging axes position around the fig.
% [6] As of r21a, boxplot attempts to adjust axis position props and that throws warning in tiledlayout.
%   - https://www.mathworks.com/matlabcentral/answers/529043
%   - https://www.mathworks.com/matlabcentral/answers/549708