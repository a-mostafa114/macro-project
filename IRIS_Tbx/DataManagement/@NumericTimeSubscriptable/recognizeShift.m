function [this, subs, sh] = recognizeShift(this, subs)
% recognizeShift  Recognize lag or lead in subscripted reference, and shift
%
% Backend [IrisToolbox] class
% No help provided

% -[IrisToolbox] for Macroeconomic Modeling
% -Copyright (c) 2007-2020 [IrisToolbox] Solutions Team

%--------------------------------------------------------------------------

if numel(subs)>=1 ...
    && strcmp(subs(1).type, '{}') ...
    && numel(subs(1).subs)==1 ...
    && isnumeric(subs(1).subs{1}) ...
    && ~isa(subs(1).subs{1}, 'DateWrapper') ...
    && isscalar(subs(1).subs{1}) ...
    && subs(1).subs{1}==round(subs(1).subs{1}) ...
    && isfinite(subs(1).subs{1})
    
    sh = subs(1).subs{1};
    this = shift(this, sh);
    subs(1) = [ ];
else
    sh = 0;
end

end%

