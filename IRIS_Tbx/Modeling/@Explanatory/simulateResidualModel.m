% simulateResidualModels  Project residuals using ResidualModels

% >=R2019b
%(
function [runningDb, innovations] = simulateResidualModel(this, runningDb, range, opt)

arguments
    this Explanatory
    runningDb {validate.databank(runningDb)}
    range {validate.properRange}

    opt.BlackoutBefore {Explanatory.validateBlackout(opt.BlackoutBefore, this)} = -Inf
    opt.SkipWhenData (1, 1) {validate.mustBeA(opt.SkipWhenData, "logical")} = false
    opt.Journal = false
end
%)
% >=R2019b

% <=R2019a
%{
function [runningDb, innovations] = simulateResidualModel(this, runningDb, range, varargin)

persistent pp
if isempty(pp)
    pp = extend.InputParser("@Explanatory/simulateResidualModel");
    addParameter(pp, "BlackoutBefore", -Inf);
    addParameter(pp, "SkipWhenData", false);
    addParameter(pp, "Journal", false);
end
opt = parse(pp, varargin{:});
%}
% <=R2019a

if isempty(range)
    return
end

journal = Journal(opt.Journal, "@Explanatory/simulateResidualModel");

%--------------------------------------------------------------------------

opt.BlackoutBefore = Explanatory.resolveBlackout(opt.BlackoutBefore);

range = double(range);
numSimulationPeriods = round(range(end) - range(1) + 1);
endHistory = dater.plus(range(1), -1);
numEquations = numel(this);

for q = 1 : numEquations
    this__ = this(q);
    if this__.IsIdentity
        if journal.IsActive
            write(journal, "Skipping " + this__.InputString);
        end
        continue
    end

    if journal.IsActive
        indent(journal, "Simulating residual model " + this__.InputString);
    end

    %
    % Retrieve history/estimation data from the input time series
    %
    residualName = this__.ResidualName;
    if isfield(runningDb, residualName)
        if isa(runningDb, "Dictionary")
            series = retrieve(runningDb, residualName);
        else
            series = runningDb.(residualName);
        end
        blackoutBefore = opt.BlackoutBefore(min(q,end));
        [data, startData] = getDataFromTo(series, blackoutBefore, range(end));
        data = data(:, :);
        if ~opt.SkipWhenData
            data(end-numSimulationPeriods+1:end, :) = NaN;
        end
    else
        data = zeros(numSimulationPeriods, 1);
    end


    %
    % Determine the total number of runs, and expand data if needed
    %
    numPages = size(data, 2);
    nv = countVariants(this__);
    numRuns = max(nv, numPages);
    if numPages==1 && numRuns>1
        data = repmat(data, 1, numRuns);
    end

    residualModel = this__.ResidualModel;
    inxMissing = isnan(data);
    if ~isempty(this__.ResidualModel) && nnz(inxMissing)>0
        numPeriods = size(data, 1);
        for v = 1 : numRuns
            if journal.IsActive && numRuns>1
                indent(journal, "Variant|Page " + sprintf("%g", v));
            end
            residualModel = update(residualModel, residualModel.Parameters(:, :, v));
            if residualModel.IsIdentity
                if journal.IsActive
                    write(journal, "Skipping identity");
                end
            else
                if journal.IsActive
                    ar = "AR=[" + sprintf("%g ", residualModel.AR) + "]";
                    ma = "MA=[" + sprintf("%g ", residualModel.MA) + "]";
                    write(journal, replace(ar, " ]", "]") + ", " + replace(ma, " ]", "]"));
                end

                %
                % Remove leading missing observations, find the last
                % available observation
                %
                first = find(~inxMissing(:, v), 1, "first");
                last = find(~inxMissing(:, v), 1, "last");
                if journal.IsActive
                    startData__ = dater.toDefaultString(dater.plus(startData, first-1));
                    endData__ = dater.toDefaultString(dater.plus(startData, last-1));
                    if journal.IsActive
                        write(journal, "Filtering data " + startData__ + ":" + endData__);
                    end
                end
                if last==numPeriods
                    continue
                end

                %
                % Convert residuals to ARMA innovations
                %
                innovations = filter(inv(residualModel), data(first:last, v));

                %
                % Add zeros on the projection horizon and convert
                % innovations back to residuals
                %
                innovations = [innovations; zeros(numPeriods-last, 1)];
                data(first:end, v) = filter(residualModel, innovations);
                if journal.IsActive
                    startSimulation__ = dater.toDefaultString(dater.plus(startData, last));
                    endSimulation__ = dater.toDefaultString(range(end));
                    if journal.IsActive
                        write(journal, "Projecting " + startSimulation__ + ":" + endSimulation__);
                    end
                end
            end
            if journal.IsActive
                deindent(journal);
            end
        end
    end


    %
    % Update the residual series in the databank
    %
    series = setData(series, startData:range(end), data);
    if isa(runningDb, "Dictionary")
        store(runningDb, residualName, series);
    else
        runningDb.(residualName) = series;
    end

    if journal.IsActive
        deindent(journal);
    end
end

end%

%
% Local Functions
%

function blackout = locallyResolveBlackout(blackout, this)
    %(
    if iscell(blackout)
        blackout = [blackout{:}];
    end
    blackout = double(blackout);
    %)
end%


%
% Local Validators
%

function locallyValidateBlackout(input, this)
end%

