% project  Project time series using a function of its own observations and exogenous inputs
%{
%}

function this = project(this, func, dates, varargin)

if isempty(this) || isempty(dates)
    return
end

%( Input parser
persistent pp
if isempty(pp)
    pp = extend.InputParser('NumericTimeSubscriptable/project');
    addRequired(pp, 'inputSeries', @(x) isa(x, 'NumericTimeSubscriptable'));
    addRequired(pp, 'function', @(x) isa(x, 'function_handle'));
    addRequired(pp, 'dates', @DateWrapper.validateProperDateInput);
    addOptional(pp, 'arguments', @(x) all(cellfun(@(y) validate.numeric(y) || isa(y, 'NumericTimeSubscriptable'), x)));
end
%)
parse(pp, this, func, dates, varargin);

%--------------------------------------------------------------------------

args = varargin;
dates = reshape(double(dates), 1, [ ]);
maxDate = max(dates);
minDate = min(dates);
[maxSh, minSh] = locallyDetermineShifts(func);

startDate = min(minDate+minSh, double(this.Start));
endDate = max(maxDate+maxSh, double(this.End));
data = getDataFromTo(this, startDate, endDate);
sizeData = size(data);
posDates = round(dates - startDate + 1);
for c = 1 : prod(sizeData(2:end))
    args__ = locallyPrepareArguments(args, c, startDate, endDate);
    for t = posDates
        data(t, c) = func(data(:, c), t, args__{:});
    end
end

this.Data = data;
this.Start = DateWrapper(startDate);
this = trim(this);

end%

%
% Local Functions
%

function [maxSh, minSh] = locallyDetermineShifts(func)
    maxSh = 0;
    minSh = 0;
    funcString = string(func2str(func));
    tokens = regexp(funcString, "\(t\s*(-\s*\d+)\s*\)", "tokens");
    if isempty(tokens)
        return
    end
    tokens = [tokens{:}];
    tokens = erase(tokens, " ");
    shifts = double(tokens);
    maxSh = max([shifts, 0]);
    minSh = min([shifts, 0]);
end%


function args__ = locallyPrepareArguments(args, c, startDate, endDate)
    args__ = args;
    for ii = 1 : numel(args__)
        if isnumeric(args__{ii})
            if ~isscalar(args{ii})
                args__{ii} = args__{ii}(c);
            end
        elseif isa(args__{ii}, 'NumericTimeSubscriptable')
            temp = getDataFromTo(args__{ii}, startDate, endDate);
            sizeTemp = size(temp);
            if prod(sizeTemp(2:end))==1
                args__{ii} = temp;
            else
                args__{ii} = temp(:, c);
            end
        end
    end
end%

