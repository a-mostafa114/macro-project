function Def = nnet( )
% nnet  [Not a public function] Default options for nnet package.
%
% Backend IRIS function.
% No help provided.

% -IRIS Macroeconomic Modeling Toolbox.
% -Copyright (c) 2007-2020 IRIS Solutions Team.

%--------------------------------------------------------------------------

Def = struct( );

Def.nnet = { ...
    'ActivationFn,Activation', 'linear', @(x) iscellstr(x) || ischar(x), ... 
	'OutputFn,Output', 'logistic', @(x) iscellstr(x) || ischar(x), ... 
	'Bias', false, @islogical, ...
    } ;

isPosInt = @(x) isnumericscalar(x) && x>0 && x==floor(x) ;

Def.estimate = { ...
    'abp', false(0), @(x) islogical(x), ...
    'learningRate,rate', 1, @(x) isnumericscalar(x,0), ...
    'NormGradient', @(x) x./norm(x,2), @isfunc, ...
    'optimset', { }, @(x) isempty(x) || isstruct(x) || (iscell(x) && iscellstr(x(1:2:end))), ...
    'Display','iter', @(x) isanystri(x,{'off','iter','final'}), ...
    'solver,optimiser,optimizer','backprop',@(x) (ischar(x) && isanystri(x,{'backprop','fmin','lsqnonlin','pso','alps'})), ...
    'Norm',@(x) norm(x,2), @isfunc, ...
    'Select', {'activation'}, @iscell, ...
    'display', 'iter', @ischar, ...
    'maxiter', 1e+4, isPosInt, ...
    'maxfunevals', 1e+5, isPosInt, ...
    'tolfun', 1e-6, @(x) isnumericscalar(x,0), ...
    'tolx', 1e-6, @(x) isnumericscalar(x,0), ...
    'recompute,recomputeObjective', 1, @(x) isPosInt(x) || isfunc(x) , ...
    'nosolution', 'penalty', true, ... % not really an option but needs to be here for irisoptim.myoptimopts
    } ;

Def.eval = {...
    'Ahead', 1, @(x) isnumericscalar(x) && x>0, ...
    'Output', 'tseries', @(x) isanystri(x,{'tseries','dbase'}), ...
    } ;

Def.plot = {...
    'Color,colour', 'blue', @(x) isanystri(x,{'activation','blue'}) ...
        || ( isnumeric(x) && ( length(x)==3 || length(x)==4 ) ) ...
    } ;

Def.prune = {...
    'EstimationOpts,Estimation', { }, @iscell, ...
    'Depth', 1, isPosInt, ... % how deep to check for anti-symmetry 
    'Method', 'correlation', @(x) isanystri(x,{'correlation'}), ...
    'Norm',@(x) norm(x,2), @isfunc, ...
    'Progress', false, @islogical, ...
    'Parallel,UseParallel', false, @islogical, ...
    'Recursive', 0, isPosInt, ...
    'Select','activation', @(x) isanystri(x,{'activation'}), ...
    } ;

end


