classdef Frequency < double
    enumeration
        INTEGER           (  0) 
        YEARLY            (  1) 
        HALFYEARLY        (  2) 
        QUARTERLY         (  4) 
        MONTHLY           ( 12) 
        WEEKLY            ( 52) 
        DAILY             (365) 
        NaF               (NaN) 

        MIN_DAILY_SERIAL  (365244)
    end


    methods
        function this = Frequency(varargin)
            this = this@double(varargin{:});
        end%




        function d = getCalendarDuration(this)
            switch this
                case Frequency.YEARLY
                    d = calendarDuration(1, 0, 0);
                case Frequency.HALFYEARLY
                    d = calendarDuration(0, 6, 0);
                case Frequency.QUARTERLY
                    d = calendarDuration(0, 3, 0);
                case Frequency.MONTHLY
                    d = calendarDuration(0, 1, 0);
                case Frequency.WEEKLY
                    d = calendarDuration(0, 0, 7);
                case Frequency.DAILY
                    d = calendarDuration(0, 0, 1);
                case Frequency.INTEGER
                    d = 1;
            end
        end%


        function daysPerPeriods = getDaysPerPeriod(this)
            switch this
                case Frequency.WEEKLY
                    daysPerPeriods = 7;
                case Frequency.DAILY
                    daysPerPeriods = 1;
                otherwise
                    daysPerPeriods = NaN;
            end
        end%


        function output = colon(a, b, varargin)
            if isa(a, 'Frequency')
                a = double(a);
            end
            if isa(b, 'Frequency')
                b = double(b);
            end
            if nargin>=3 && isa(varargin{1}, 'Frequency')
                varargin{1} = double(varargin{1});
            end
            output = colon(a, b, varargin{:});
        end%


        function datetimeObj = datetime(this, serial, varargin)
            datetimeObj = dater.matlabFromSerial(double(this), double(serial), varargin{:});
            if isequaln(this, Frequency.NaF)
                datetimeObj = NaT(size(serial));
                return
            end
            year = zeros(size(serial));
            month = zeros(size(serial));
            day = zeros(size(serial));
            indexInf = isinf(serial);
            [year(~indexInf), month(~indexInf), day(~indexInf)] = ...
                dater.ymdFromSerial(this, serial(~indexInf), varargin{:});
            if this==Frequency.WEEKLY
                day(~indexInf) = day(~indexInf) - 3; % Return Monday, not Thursday, for display
            end
            year(indexInf) = serial(indexInf);
            datetimeObj = datetime(year, month, day, 'Format', dater.getFormatForMatlab(this));
        end%


        function [durationObj, halfDurationObj] = duration(this)
            switch this
            case Frequency.YEARLY
                durationObj = calyears(1);
                halfDurationObj = calmonths(6);
            case Frequency.HALFYEARLY
                durationObj = calmonths(6);
                halfDurationObj = calmonths(3);
            case Frequency.QUARTERLY
                durationObj = calquarters(1);
                halfDurationObj = caldays(45);
            case Frequency.MONTHLY
                durationObj = calmonths(1);
                halfDurationObj = caldays(15);
            case Frequency.WEEKLY
                durationObj = calweeks(1);
                halfDurationObj = days(3.5);
            case Frequency.DAILY
                durationObj = caldays(1);
                halfDurationObj = days(0.5);
            otherwise
                durationObj = NaN;
                halfDurationObj = NaN;
            end
        end%


        function flag = isnaf(this)
            flag = isnan(this);
        end%


        function [highExtStartSerial, highExtEndSerial, lowStartSerial, lowEndSerial, ixHighInLowBins] = ...
                aggregateRange(highFreq, highStartSerial, highEndSerial, lowFreq)
            [year1, month1, day1] = dater.ymdFromSerial(highFreq, highStartSerial, 'Start');
            lowStartSerial = dater.serialFromYmd(lowFreq, year1, month1, day1);
            [year2, month2, day2] = dater.ymdFromSerial(lowFreq, lowStartSerial, 'Start');
            highExtStartSerial = dater.serialFromYmd(highFreq, year2, month2, day2);

            [year3, month3, day3] = dater.ymdFromSerial(highFreq, highEndSerial, 'End');
            lowEndSerial = dater.serialFromYmd(lowFreq, year3, month3, day3);
            [year4, month4, day4] = dater.ymdFromSerial(lowFreq, lowEndSerial, 'End');
            highExtEndSerial = dater.serialFromYmd(highFreq, year4, month4, day4);

            highExtRangeSerial = highExtStartSerial : highExtEndSerial;
            [year5, month5, day5] = dater.ymdFromSerial(highFreq, highExtRangeSerial, 'middle');
            lowRangeSerial = dater.serialFromYmd(lowFreq, year5, month5, day5);

            uniqueLowRangeSerial = unique(lowRangeSerial, 'stable');
            ixHighInLowBins = cell(size(uniqueLowRangeSerial));
            for ithBin = 1 : numel(ixHighInLowBins)
                ixHighInLowBins{ithBin} = lowRangeSerial==uniqueLowRangeSerial(ithBin);
            end

            lowStartSerial = uniqueLowRangeSerial(1);
            lowEndSerial = uniqueLowRangeSerial(end);
        end%


        function c = cellstr(this)
            c = arrayfun(@(x) char(x), this, 'UniformOutput', false);
        end%
    end




    methods (Static)
        function period = month2period(this, month)
            %(
            this = double(this);
            switch this
                case 1
                    period = ones(size(month));
                case 2
                    period = floor(double(month+5)/6);
                case 4
                    period = floor(double(month+2)/3);
                case 12
                    period = month;
                otherwise
                    perios = nan(size(month));
            end
            %)
        end%


        function ppy = getPeriodsPerYear(this)
            %(
            this = double(this);
            if any(this==[1, 2, 4, 12])
                ppy = this;
            else
                ppy = NaN;
            end
            %)
        end%


        function serial = serialize(this, varargin)
            %(
            this = double(this);
            switch this
                case 0
                    serial = round(varargin{1});
                case {1, 2, 4, 12}
                    ppy = Frequency.getPeriodsPerYear(this);
                    year = round(varargin{1});
                    period = 1;
                    if numel(varargin)>1
                        period = round(varargin{2});
                    end
                    serial = round(ppy*year + period - 1);
                case 52
                    [year, month, day] = varargin{1:3};
                    dailySerial = floor(datenum(year, month, day));
                    sh = Frequency.shiftToThursday(dailySerial);
                    thursdayDailySerial = round(datenum(varargin{:}) + sh); % Serial of Thursday in the same week
                    fridayDailySerial = thursdayDailySerial + 1;
                    serial = round(fridayDailySerial/7); % Fridays are divisible by 7
                    serial = serial - 1; % Subtract 1 to have the first entire week in year 0 numbered 1
                case 365
                    [year, month, day] = varargin{1:3};
                    dailySerial = floor(datenum(year, month, day));
                    serial = dailySerial;
                otherwise
                    serial = NaN;
            end
            %)
        end%


        function [year, period] = serial2yp(this, serial)
            %(
            this = double(this);
            switch this
                case 0
                    year = nan(size(serial));
                    period = Frequency.deserialize(this, serial);
                case {1, 2, 4, 12}
                    [year, period] = Frequency.deserialize(this, serial);
                case 52
                    year = Frequency.deserialize(this, serial);
                    firstThursdayOfYear = Frequency.firstThursdayOfYear(year);
                    fwy = Frequency.serialize(this, year, 1, firstThursdayOfYear);
                    period = round(serial - fwy + 1);
                case 365
                    [year, month, day] = Frequency.deserialize(this, serial);
                    startOfYear = floor(datenum(double(year), 1, 1));
                    period = round(serial - startOfYear + 1);
                otherwise
                    year = nan(size(serial));
                    period = nan(size(serial));
            end
            %)
        end%


        function [year, month, day] = serial2ymd(this, serial, position)
            %(
            if nargin>=3
                position = lower(extractBefore(string(position), 2));
                if ~any(position==["s", "m", "e"])
                    position = "s";
                end
            else
                position = "s";
            end
            this = double(this);
            switch this
                case 1
                    year = Frequency.deserialize(this, serial);
                    switch position
                        case "s" % Start of period
                            month = ones(size(year));
                            day = ones(size(year));
                        case "m" % Middle of period
                            month = repmat(6, size(year));
                            day = repmat(30, size(year));
                        case "e" % End of period
                            month = repmat(12, size(year));
                            day = repmat(31, size(year));
                    end
                case 2
                    [year, halfyear] = Frequency.deserialize(this, serial);
                    month = 6*(halfyear-1);
                    switch position
                        case "s" % Start of period
                            month = month + 1;
                            day = ones(size(year));
                        case "m" % Middle of period
                            month = month + 4;
                            day = ones(size(year));
                        case "e" % End of period
                            month = month + 6;
                            day = eomday(year, month);
                    end 
                case 4
                    [year, quarter] = Frequency.deserialize(this, serial);
                    month = 3*(quarter-1);
                    switch position
                        case "s" % Start of period
                            month = month + 1;
                            day = ones(size(year));
                        case "m" % Middle of period
                            month = month + 2;
                            day = repmat(15, size(year));
                        case "e" % End of period
                            month = month + 3;
                            day = eomday(year, month);
                    end
                case 12
                    [year, month] = Frequency.deserialize(this, serial);
                    switch position
                        case "s" % Start of period
                            day = ones(size(year));
                        case "m" % Middle of period
                            day = repmat(15, size(year));
                        case "e" % End of period
                            day = eomday(year, month);
                    end
                case 52
                    [year, month, day] = Frequency.deserialize(this, serial);
                    switch position
                        case "s" % Start of period
                            day = day - 3; % Return Monday
                        case "m" % Middle of period
                            day = day + 0; % Return Thursday
                        case "e" % End of period
                            day = day + 3; % Return Sunday
                    end
                case 365
                    [year, month, day] = Frequency.deserialize(this, serial);
                otherwise
                    year = nan(size(serial));
                    month = nan(size(serial));
                    day = nan(size(serial));
            end
            year = round(year);
            month = round(month);
            day = round(day);
            %)
        end%


        function varargout = deserialize(this, serial)
            %(
            switch double(this)
                case 0
                    varargout = { round(serial) };
                case {1, 2, 4, 12}
                    ppy = Frequency.getPeriodsPerYear(this);
                    year = floor(serial/ppy);
                    period = round(serial - ppy*year + 1);
                    varargout = { year, period };
                case 52
                    dailySerial = round((serial + 1)*7 - 1); % 7-multiples of weekly serials are Fridays, return Thursday.
                    [year, month, day] = datevec(dailySerial);
                    varargout = { round(year), round(month), round(day) };
                case 365
                    [year, month, day] = datevec(serial);
                    varargout = { round(year), round(month), round(day) };
                otherwise
                    varargout = { nan(size(serial)) };
            end
            %)
        end%


        function d = getXLimMarginCalendarDuration(this)
            %(
            switch double(this)
                case 1
                    d = calendarDuration(0, 6, 0);
                case 2
                    d = calendarDuration(0, 3, 0);
                case 4
                    d = calendarDuration(0, 1, 15);
                case 12
                    d = calendarDuration(0, 0, 15);
                case 52
                    d = calendarDuration(0, 0, 3);
                case 365
                    d = calendarDuration(0, 0, 1);
                case 0
                    d = 0.5;
                otherwise
                    d = NaN;
            end
            %)
        end%
        

        function c = toChar(freq)
            if ~isa(freq, 'Frequency')
                freq = Frequency(freq);
            end
            c = char(freq);
        end%


        function s = toString(freq)
            if ~isa(freq, 'Frequency')
                freq = Frequency(freq);
            end
            s = string(freq);
        end%


        function c = toCellstr(freq)
            if ~isa(freq, 'Frequency')
                freq = Frequency(freq);
            end
            c = cellstr(freq);
        end%


        function sh = shiftToThursday(dailySerial)
            sh = zeros(size(dailySerial));
            dayOfWeek = weekday(dailySerial);
            sh(dayOfWeek==1) = -3; % Sunday
            sh(dayOfWeek==2) = +3; % Monday
            sh(dayOfWeek==3) = +2; % Tuesday
            sh(dayOfWeek==4) = +1; % Wednesday
            sh(dayOfWeek==5) = +0; % Thursday
            sh(dayOfWeek==6) = -1; % Friday
            sh(dayOfWeek==7) = -2; % Saturday
        end%


        function day = firstThursdayOfYear(year)
            % Day of first Thursday in January.
            s = floor(datenum(double(year), 1, 1));
            temp = datenum(double(year), 1, 1);
            temp = weekday(temp);
            ixSun = temp==1;
            ixMon = temp==2;
            ixTue = temp==3;
            ixWed = temp==4;
            ixFri = temp==6;
            ixSat = temp==7;
            offset = zeros(size(temp));
            offset(ixSun) = 4;
            offset(ixMon) = 3;
            offset(ixTue) = 2;
            offset(ixWed) = 1;
            offset(ixFri) = 6;
            offset(ixSat) = 5;
            day = 1 + offset;
        end%


        function x = empty(varargin)
            x = repmat(Frequency.NaF, varargin{:});
        end%


        function this = fromString(string)
            if ~ischar(string) && ~(isa(string, 'string') && isscalar(string))
                throw( exception.Base('Frequency:InvalidConversionFromString', 'error') );
            end
            switch upper(char(string))
                case {'INTEGER', 'II', 'I'}
                    this = Frequency.INTEGER;
                case {'DAILY', 'DAY', 'DD', 'D'}
                    this = Frequency.DAILY;
                case {'WEEKLY', 'WEEK', 'WW', 'W'}
                    this = Frequency.WEEKLY;
                case {'MONTHLY', 'MONTH', 'MM', 'M'}
                    this = Frequency.MONTHLY;
                case {'QUARTERLY', 'QUARTER', 'QQ', 'Q'}
                    this = Frequency.QUARTERLY;
                case {'HALFYEARLY', 'HALFYEAR', 'SEMIANNUAL', 'SEMIANNUALLY', 'HH', 'H'}
                    this = Frequency.HALFYEARLY;
                case {'YEARLY', 'YEAR', 'ANNUAL', 'ANNUALLY', 'YY', 'Y', 'A'}
                    this = Frequency.YEARLY;
                case {'NAF', 'NAN', 'N'}
                    this = Frequency.NaF;
                otherwise
                    throw( exception.Base('Frequency:InvalidConversionFromString', 'error') );
            end
        end%


        function [flag, this] = validateFrequency(input)
            if isa(input, 'Frequency')
                flag = true;
                this = input;
                return
            end
            if isnumeric(input)
                this = Frequency.NaF;
                try
                    this = Frequency.fromNumeric(input);
                    flag = true;
                    return
                catch
                    flag = false;
                    return
                end
            elseif ischar(input) || (isa(input, 'string') && isscalar(input))
                this = Frequency.NaF;
                try
                    this = Frequency.fromString(input);
                    flag = true;
                    return
                catch
                    flag = false;
                    return
                end
            end
            flag = false;
        end%


        function [flag, this] = validateProperFrequency(input)
            [flag, this] = Frequency.validateFrequency(input);
            if flag
                flag = ~isempty(this) && ~isnan(this) && this~=Frequency.INTEGER;
            end
        end%


        function this = fromNumeric(input)
            try
                this = Frequency(input);
            catch
                throw( exception.Base('Frequency:InvalidConversionFromNumeric', 'error') );
            end
        end%


        function letter = toLetter(freq)
            freqLetters = iris.get('FreqLetters'); 
            letter = repmat("", size(freq));
            letter(freq==Frequency.YEARLY) = freqLetters(1);
            letter(freq==Frequency.HALFYEARLY) = freqLetters(2);
            letter(freq==Frequency.QUARTERLY) = freqLetters(3);
            letter(freq==Frequency.MONTHLY) = freqLetters(4);
            letter(freq==Frequency.WEEKLY) = freqLetters(5);
            letter(freq==Frequency.DAILY) = "D";
            letter(freq==Frequency.INTEGER) = "I";
        end%


        function flag = sameFrequency(freq1, freq2, varargin)
            if isempty(freq1)
                flag = true;
                return
            end
            if isscalar(freq1) && (nargin<2 || isempty(freq2))
                flag = true;
                return
            end
            if nargin==1 || isempty(freq2)
                if isscalar(freq1)
                    flag = true;
                    return
                end
                freq1 = double(freq1);
                freq2 = freq1(2:end);
                freq1 = freq1(1);
            end
            if all(double(freq1)==double(freq2))
                flag = true;
                return
            end
            flag = false;
        end%


        function c = toFredLetter(this)
            %(
            switch this
                case Frequency.YEARLY
                    c = "A";
                case Frequency.HALFYEARLY
                    c = "SA";
                case Frequency.QUARTERLY
                    c = "Q";
                case Frequency.MONTHLY
                    c = "M";
                otherwise
                    exception.error([ 
                        "Frequency:InvalidFredFrequency" 
                        "This is not a valid Fred frequency: %s" 
                        "Fred frequency needs to be one of {YEARLY, HALFYEARLY, QUARTERY, MONTHLY}." 
                    ], this);
            end
            %)
        end%


        function c = toIMFLetter(this)
            %(
            switch this
                case Frequency.YEARLY
                    c = "A";
                case Frequency.QUARTERLY
                    c = "Q";
                case Frequency.MONTHLY
                    c = "M";
                otherwise
                    exception.error([ 
                        "Frequency:InvalidIMFFrequency" 
                        "This is not a valid IMF frequency: %s" 
                        "IMF frequency needs to be one of {YEARLY, QUARTERY, MONTHLY}." 
                    ], this);
            end
            %)
        end%
    end
end

