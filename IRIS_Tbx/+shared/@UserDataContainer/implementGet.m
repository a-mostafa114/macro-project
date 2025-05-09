function [answ, flag, query] = implementGet(this, query, varargin)
% implementGet  Implement get method for shared.UserDataContainer objects
%
% Backend IRIS function
% No help provided

% -IRIS Macroeconomic Modeling Toolbox
% -Copyright (c) 2007-2020 IRIS Solutions Team

%--------------------------------------------------------------------------

answ = [ ];
flag = true;
if any(strcmpi(query, {'BaseYear', 'TOrigin'}))
    answ = this.BaseYear;
    if isequal(answ, @config) || isempty(answ)
        answ = iris.get('BaseYear');
    end
else
    flag = false;
end

end%

