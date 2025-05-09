function varargout = testSuite(varargin)
% -IRIS Macroeconomic Modeling Toolbox.
% -Copyright (c) 2007-2020 IRIS Solutions Team.

persistent X %#ok<*USENS>

if (nargin == 0 && nargout == 0) || isempty(X)
    X = doPopulate( );
end

if nargout == 0
    return
elseif nargin == 0
    varargout{1} = X;
    return
end

varargout = cell(size(varargin)) ;
for ii = 1:numel(varargin)
    varargout{ii} = X.(varargin{ii}) ;
end

    function X = doPopulate( )
        X = struct( ) ;
        
        %% local optimization test functions
        % rosenbock
        a = 1 ;
        b = 10 ;
        rosenbock = @(x,y,a,b) (a-x)^2 + b*(y-x^2)^2 ;
        X.rosenbock = irisoptim.testFn(...
            @(x) rosenbock(x(1),x(2),a,b),...
            'rosenbock',[a,a^2],'x0=',[.5,.5],...
            'type=','local',...
            'notes=','parabolic valley') ;
                
        % beale
        X.beale = irisoptim.testFn(...
            @(x) (1.5-x(1)+x(1)*x(2))^2 ...
            + (2.25-x(1)+x(1)*x(2)^2)^2 ...
            + (2.625-x(1)+x(1)*x(2)^3)^2, ...
            'beale',[3;0.5],'x0=',[2;-2], ...
            'type=','local') ;
        
        % matyas
        X.matyas = irisoptim.testFn(...
            @(x) 0.26*(x(1)^2+x(2)^2)-0.48*x(1)*x(2), ...
            'matyas',[0;0],'type=','local', ...
            'x0=',[-3;3]) ;
        
        % mccormick
        X.mccormick = irisoptim.testFn(...
            @(x) sin(x(1)+x(2)) + (x(1)-x(2))^2 - 1.5*x(1) + 2.5*x(2) + 1, ...
            'mccormick',[-0.54719;-1.54719],'type=','local',...
            'x0=',[-3;3]) ; 
        
        % styblinskitang
        X.styblinskitang = irisoptim.testFn(...
            ( sum(x.^4)-sum(16*x.^2)+5*sum(x) )/2, ...
            'styblinskitang',@(d) -2.90354*ones(d,1), ...
            'type=','local') ; 
        
        % booth
        X.booth = irisoptim.testFn(...
            (x(1)+2*x(2)-7)^2 + (2*x(1)+x(2)-5)^2, ...
            'booth',[1;3], ...
            'x0=', [-1; -1.5], ...
            'type=','local') ;
        
        % michalewicz
        steepness = 10 ;
        d = 2 ;
        X.michalewicz = irisoptim.testFn(...
            @(x) -sum( sin(x).*sin(transpose(1:d).*x.^2/pi).^steepness ), ...
            'michalewicz',[2.18531185370279; 1.57078454451371], ...
            'type=','local') ;
        
        %% global optimization test functions
        
        %% 1d root finding test functions
    end

end