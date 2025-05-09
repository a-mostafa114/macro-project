classdef graphobj < report.genericobj
    
    
    
    
    methods
        function this = graphobj(varargin)
            this = this@report.genericobj(varargin{:});
            this.childof = {'figure'};
            this.default = [this.default, { ...
                'axesoptions', { }, @(x) iscell(x) && iscellstr(x(1:2:end)), true, ...
                'rhsaxesoptions', { }, @(x) iscell(x) && iscellstr(x(1:2:end)), true, ...
                'DateTick', @auto, @(x) isequal(x, @auto) || isnumeric(x), true, ...
                'grid', @auto, @(x) isequal(x, @auto) || islogicalscalar(x), true, ...
                'highlight', [ ], @(x) isnumeric(x) ...
                || (iscell(x) && all(cellfun(@isnumeric, x))), true, ... Obsolete, use highlight object.
                'legend', false, @(x) islogical(x) || isnumeric(x), true, ...
                'legendlocation', 'NorthEast', @ischar, true, ...
                'legendoptions', { }, @(x) iscell(x) && iscellstr(x(1:2:end)), true, ...
                'preprocess', '', @(x) isempty(x) || ischar(x), true, ...
                'postprocess', '', @(x) isempty(x) || ischar(x), true, ...
                'range', Inf, @isnumeric, true, ...
                'style', [ ], @(x) isempty(x) || isstruct(x), true, ...
                'tight', @auto, @(x) isequal(x, @auto) || islogicalscalar(x), true, ...
                'titleoptions', { }, @(x) iscell(x) && iscellstr(x(1:2:end)), true, ...
                'xlabel', '', @ischar, true, ...
                'ylabel', '', @ischar, true, ...
                'zlabel', '', @ischar, true, ...
                'zeroline', false, ...
                @(x) islogical(x) || (iscell(x) && iscellstr(x(1:2:end))), ...
                true, ...
                ...
                ... Date format options
                ...---------------------
                'dateformat', @config, @iris.Configuration.validateDateFormat, true, ...
                'freqletters', @config, @iris.Configuration.validateFreqLetters, true, ...
                'months', @config, @iris.Configuration.validateMonths, true, ...
                'standinmonth', @config, @iris.Configuration.validateConversionMonth, true, ...
                }];
        end        
        
        
        
        
        function [this, varargin] = specargin(this, varargin)
        end
        
        
        
        
        function this = setoptions(this, varargin)
            this = setoptions@report.genericobj(this, varargin{:});
            % Remove equal signs from name-value pairs of options passed into Matlab
            % functions.
            list = {'axes', 'rhsaxes', 'legend', 'title'};
            for i = 1 : length(list)
                name = [list{i}, 'options'];
                if ~isempty(this.options.(name))
                    this.options.(name)(1:2:end) ...
                        = strrep(this.options.(name)(1:2:end), '=', '');
                end
            end
        end        
        
        
        
        
        function ax = subplot(this, R, C, I, varargin) %#ok<INUSL>
            ax = subplot(R, C, I, varargin{:});
        end
        
        
        
        
        varargout = plot(varargin)
    end
    
end
