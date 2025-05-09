function [deriv, ixNanDeriv] = diffFirstOrder(this, eqSelect, variantRequested, opt)
% diffFirstOrder  Calculate first-order derivatives of equations
%
% Backend IRIS function
% No help provided

% -IRIS Macroeconomic Modeling Toolbox
% -Copyright (c) 2007-2020 IRIS Solutions Team

TYPE = @int8;

isNanDeriv = nargout > 2;

%--------------------------------------------------------------------------

% Copy last computed derivatives.
deriv = this.LastSystem.Deriv;

asgn = this.Variant.Values(:, :, variantRequested);
nName = numel(this.Quantity);
nEqtn = numel(this.Equation);
ixy = this.Quantity.Type==TYPE(1);
ixx = this.Quantity.Type==TYPE(2);
ixe = this.Quantity.Type==TYPE(31) | this.Quantity.Type==TYPE(32);
ixyxe = ixy | ixx | ixe;
ixp = this.Quantity.Type==TYPE(4);
ixm = this.Equation.Type==TYPE(1);
ixt = this.Equation.Type==TYPE(2);
inxOfLog = this.Quantity.InxLog;
eqSelect(~ixm & ~ixt) = false;

ixNanDeriv = false(1, nEqtn);

% Prepare 3D occur array limited to occurences of variables and shocks in
% measurement and transition equations.
nsh = this.Incidence.Dynamic.NumOfShifts;

if any(eqSelect)    
    nyxe = sum(ixyxe);
    sh0 = this.Incidence.Dynamic.PosOfZeroShift;
    if opt.Symbolic
        ixSymb = ~cellfun(@isempty, this.Gradient.Dynamic(1, :));
    else
        ixSymb = false(1, nEqtn);
    end
    ixSymb = ixSymb & eqSelect;
    ixNum = ~ixSymb & eqSelect;
    
    if any(ixSymb)
        % Symbolic derivatives.
        symbDeriv( );
    end
    if any(ixNum)
        % Numerical derivatives.
        numDeriv( );
    end
    
    % Reset the add-factors in nonlinear equations to 1.
    tempEye = -eye(sum(this.Equation.Type<=2));
    deriv.n(eqSelect, :) = tempEye(eqSelect, this.Equation.IxHash);
    
    % Normalize derivatives by largest number in nonlinear models.
    if ~this.IsLinear && opt.Normalize
        for iEq = find(eqSelect)
            ix = deriv.f(iEq, :)~=0;
            if any(ix)
                norm = max(abs(deriv.f(iEq, ix)));
                deriv.f(iEq, ix) = deriv.f(iEq, ix) / norm;
                deriv.n(iEq, :) = deriv.n(iEq, :) / norm;
            end
        end
    end
end

return


    function numDeriv( )
        minT = 1 - sh0;
        maxT = nsh - sh0;
        tVec = minT : maxT;
        
        if this.IsLinear
            init = zeros(nName, 1);
            init(ixp) = real(asgn(ixp));
            init = repmat(init, 1, nsh);
            h = ones(size(init));
        else
            isDelog = false;
            init = createTrendArray(this, variantRequested, isDelog, 1:nName, tVec);
            diffStep = this.Tolerance.DiffStep;
            maxInitOr1 = init;
            maxInitOr1(init<1) = 1;
            h = diffStep*maxInitOr1;
        end
        
        xPlus = init + h;
        xMinus = init - h;
        % Any imag parts in `xPlus` and `xMinus` should cancel; `real( )` does no
        % harm here therefore.
        step = real(xPlus - xMinus);
        
        % Delog log-plus variables.
        if any(inxOfLog)
            init(inxOfLog, :) = real(exp( init(inxOfLog,:) ));
            xPlus(inxOfLog, :) = real(exp( xPlus(inxOfLog,:) ));
            xMinus(inxOfLog, :) = real(exp( xMinus(inxOfLog,:) ));
        end
        
        % References to steady levels; can be used only in nonlinear setup.
        if this.IsLinear
            L = [ ];
        else
            L = init;
        end
        
        for iiEq = find(ixNum)
            eqtn = vectorize( this.Equation.Dynamic{iiEq} );
            fn = str2func([this.PREAMBLE_DYNAMIC, '[', eqtn, ']']);
            
            % Get incidence of variables in this equation.
            nm = reshape(real(this.Gradient.Dynamic{2, iiEq}), 1, [ ]);
            sh = sh0 + reshape(imag(this.Gradient.Dynamic{2, iiEq}), 1, [ ]);

            % Total number of derivatives to be computed in this equation.
            n = numel(nm);
            value = zeros(1, n);
            for ii = 1 : n
                iNm = nm(ii);
                iSh = sh(ii);
                gridMinus = init;
                gridPlus = init;
                gridMinus(iNm, iSh) = xMinus(iNm, iSh);
                gridPlus(iNm, iSh) = xPlus(iNm, iSh);
                fMinus = fn(gridMinus, sh0, L);
                fPlus =  fn(gridPlus, sh0, L);
                value(ii) = (fPlus-fMinus) / step(nm(ii), sh(ii));
            end
            
            % Constant in linear models.
            if this.IsLinear
                deriv.c(iiEq) = fn(init, sh0, L);
            end
            
            % Assign values to the array of derivatives.
            ix = (sh-1)*nyxe + nm;
            deriv.f(iiEq, ix) = value;
            
            % Check for NaN derivatives.
            if isNanDeriv && any( ~isfinite(value) )
                ixNanDeriv(iiEq) = true;
            end
        end
    end

    
    function symbDeriv( )
        if this.IsLinear
            x = zeros(nName, 1);
            x(inxOfLog) = 1;
            x(ixp) = real(asgn(ixp));
            x = repmat(x, 1, nsh);
            % References to steady-state levels.
            L = [ ];
        else
            isDelog = true;
            x = createTrendArray(this, variantRequested, isDelog);
            % References to steady-state levels
            L = x;
        end
   
        for iiEq = find(ixSymb)
            % Get incidence of variables in this equation
            nm = reshape(real(this.Gradient.Dynamic{2, iiEq}), 1, [ ]);
            sh = sh0 + reshape(imag(this.Gradient.Dynamic{2, iiEq}), 1, [ ]);

            % Log derivatives need to be multiplied by x 
            %
            % Log-plus and log-minus variables are treated the same way
            % because
            %
            % df(x)/dlog(xm) = df(x)/d(x) * d(x)/d(xm) * d(xm)/dlog(xm)
            %                = df(x)/d(x) * (-1) * xm = df(x)/d(x) * (-1) * (-1)*x
            %                = df(x)/d(x) * x
            logMult = [ ];
            if any(inxOfLog(nm))
                logMult = ones(size(nm));
                for ii = find( inxOfLog(nm) )
                    logMult(ii) = x(nm(ii), sh(ii));
                end
            end
            
            % Evaluate all derivatives at once.
            value = this.Gradient.Dynamic{1, iiEq}(x, sh0, L);
            value = reshape(value, 1, [ ]);
            
            % Multiply derivatives wrt to log variables by x.
            if ~isempty(logMult)
                value = value .* logMult;
            end

            % Assign values to the array of derivatives.
            ix = (sh-1)*nyxe + nm;
            deriv.f(iiEq, ix) = value;
            
            % Check for NaN derivatives.
            if isNanDeriv && any( ~isfinite(value) )
                ixNanDeriv(iiEq) = true;
            end
        end
        
        % Evaluate all equations at x=0, log(x)=0 to get constant terms.
        if this.IsLinear
            fn = str2func([this.PREAMBLE_DYNAMIC, '[', this.Equation.Dynamic{ixSymb}, ']', ]);
            deriv.c(ixSymb) = fn(x, sh0, L);
        end
    end
end
