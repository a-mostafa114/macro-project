function [F, Pe, V, delta, PDelta, sampleCov, this] ...
    = kalmanFilterRegOutp(this, regOutp, xRange, likOpt, opt);
% kalmanFilterRegOutp  Postprocess regular (non-hdata) output arguments from the Kalman filter or FD lik.
%
% Backend [IrisToolbox] method
% No help provided

% -[IrisToolbox] for Macroeconomic Modeling
% -Copyright (c) 2007-2020 [IrisToolbox] Solutions Team

try
    isNamedMat = strcmpi(opt.MatrixFormat, 'namedmat');
catch
    isNamedMat = false;
end

TIME_SERIES_CONSTRUCTOR = iris.get('DefaultTimeSeriesConstructor');
TIME_SERIES_TEMPLATE = TIME_SERIES_CONSTRUCTOR( );
TYPE = @int8;

%--------------------------------------------------------------------------

ixy = this.Quantity.Type==TYPE(1);
ixe = this.Quantity.Type==TYPE(31) | this.Quantity.Type==TYPE(32);

startDate = xRange(1);

F = [ ];
if isfield(regOutp, 'F')
    F = TIME_SERIES_TEMPLATE;
    F = replace(F, permute(regOutp.F, [3, 1, 2, 4]), startDate);
end

Pe = [ ];
if isfield(regOutp, 'Pe')
    Pe = struct( );
    for iName = find(ixy)
        name = this.Quantity.Name{iName};
        data = permute(regOutp.Pe(iName, :, :), [2, 3, 1]);
        if this.Quantity.IxLog(iName)
            data = real(exp(data));
        end
        Pe.(name) = TIME_SERIES_TEMPLATE;
        Pe.(name) = replace(Pe.(name), data, startDate, name);
    end
end

V = [ ];
if isfield(regOutp, 'V')
    V = regOutp.V;
end

% Update out-of-lik parameters in the model object.
delta = struct( );
lsDelta = this.Quantity.Name(likOpt.OutOfLik);
if isfield(regOutp, 'Delta')
    for i = 1 : length(likOpt.OutOfLik)
        name = lsDelta{i};
        posQty = likOpt.OutOfLik(i);
        this.Variant.Values(:, posQty, :) = regOutp.Delta(i, :);
        delta.(name) = regOutp.Delta(i, :);
    end
end

PDelta = [ ];
if isfield(regOutp, 'PDelta')
    PDelta = regOutp.PDelta;
    if isNamedMat
        PDelta = namedmat(PDelta, lsDelta, lsDelta);
    end
end

sampleCov = [ ];
if isfield(regOutp, 'SampleCov')
    sampleCov = regOutp.SampleCov;
    if isNamedMat
        eList = this.Quantity.Name(ixe);
        sampleCov = namedmat(sampleCov, eList, eList);
    end
end

% Update the std parameters in the model object.
if likOpt.Relative && nargout>6
    ne = sum(ixe);
    nv = length(this);
    se = sqrt(V);
    for v = 1 : nv
        this.Variant.StdCorr(:, 1:ne, v) = this.Variant.StdCorr(:, 1:ne, v)*se(v);
    end
    % Refresh dynamic links after we change std deviations because std devs are
    % allowed in dynamic links.
    if any(this.Link)
        this = refresh(this);
    end
end

end
