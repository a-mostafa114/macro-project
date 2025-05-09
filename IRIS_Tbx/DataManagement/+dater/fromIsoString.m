% dater.fromIsoString  Convert ISO string to numeric dateCode
%{
% Syntax
%--------------------------------------------------------------------------
%
%     output = dater.fromIsoString(freq, isoString)
%
%
% Input Arguments
%--------------------------------------------------------------------------
%
% __``__ [ ]
%
%>    Description
%
%
% Output Arguments
%--------------------------------------------------------------------------
%
% __``__ [ ]
%
%>    Description
%
%
% Options
%--------------------------------------------------------------------------
%
%
% __`=`__ [ | ]
%
%>    Description
%
%
% Description
%--------------------------------------------------------------------------
%
%
% Example
%--------------------------------------------------------------------------
%
%}

% -[IrisToolbox] for Macroeconomic Modeling
% -Copyright (c) 2007-2020 [IrisToolbox] Solutions Team

function dateCode = fromIsoString(freq, isoDate)

freq = round(double(freq));
isoDate = string(isoDate);

if isequal(freq, 0)
    dateCode = double(isoDate);
    return
end

reshapeOutput = size(isoDate);
isoDate = reshape(isoDate, 1, [ ]);
[isoDate, inxMissing] = locallyFixIsoDate(isoDate);

[year, month, day] = locallyGetYearMonthDay(isoDate);

serial = dater.serialFromYmd(freq, year, month, day);

dateCode = nan(size(inxMissing));
dateCode(~inxMissing) = dater.fromSerial(freq, serial);
dateCode = reshape(dateCode, reshapeOutput);

end%

%
% Local Functions
%

function [isoDate, inxMissing] = locallyFixIsoDate(isoDate)
    inxMissing = ismissing(isoDate);
    isoDate(inxMissing) = [ ];
    lenIsoDate = strlength(isoDate);
    inx10 = lenIsoDate>10;
    if any(inx10)
        isoDate(inx10) = extractBefore(isoDate(inx10), 11);
    end
    inx7 = lenIsoDate==7;
    if any(inx7)
        isoDate(inx7) = isoDate(inx7) + "-01";
    end
    inx4 = lenIsoDate==4;
    if any(inx4)
        isoDate(inx4) = isoDate(inx4) + "-01-01";
    end
end%


function [year, month, day] = locallyGetYearMonthDay(isoDate)
    [year, month, day] = textual.split(isoDate, "-");
    year = double(year);
    month = double(month);
    day = double(day);
end%




%
% Unit Tests
%
%{
##### SOURCE BEGIN #####
% saveAs=dater/fromIsoStringUnitTest.m

testCase = matlab.unittest.FunctionTestCase.fromFunction(@(x)x);


%% Test Full String
    assertEqual(testCase, dater.fromIsoString(Frequency.DAILY, "2020-05-15"), dater.dd(2020,05,15));
    assertEqual(testCase, dater.fromIsoString(Frequency.WEEKLY, "2020-05-15"), dater.ww(2020,05,15));
    assertEqual(testCase, dater.fromIsoString(Frequency.MONTHLY, "2020-05-15"), dater.mm(2020,05));
    assertEqual(testCase, dater.fromIsoString(Frequency.QUARTERLY, "2020-05-15"), dater.qq(2020,2));
    assertEqual(testCase, dater.fromIsoString(Frequency.YEARLY, "2020-05-15"), dater.yy(2020));


%% Test Year Month String
    assertEqual(testCase, dater.fromIsoString(Frequency.DAILY, "2020-05"), dater.dd(2020,05,01));
    assertEqual(testCase, dater.fromIsoString(Frequency.WEEKLY, "2020-05"), dater.ww(2020,05,01));
    assertEqual(testCase, dater.fromIsoString(Frequency.MONTHLY, "2020-05"), dater.mm(2020,05));
    assertEqual(testCase, dater.fromIsoString(Frequency.QUARTERLY, "2020-05"), dater.qq(2020,2));
    assertEqual(testCase, dater.fromIsoString(Frequency.YEARLY, "2020-05"), dater.yy(2020));
    

%% Test Year String
    assertEqual(testCase, dater.fromIsoString(Frequency.DAILY, "2020"), dater.dd(2020,01,01));
    assertEqual(testCase, dater.fromIsoString(Frequency.WEEKLY, "2020"), dater.ww(2020,01,01));
    assertEqual(testCase, dater.fromIsoString(Frequency.MONTHLY, "2020"), dater.mm(2020,01));
    assertEqual(testCase, dater.fromIsoString(Frequency.QUARTERLY, "2020"), dater.qq(2020,1));
    assertEqual(testCase, dater.fromIsoString(Frequency.YEARLY, "2020"), dater.yy(2020));
    
##### SOURCE END #####
%}
