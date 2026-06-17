function datetime_out = convert_datetime(str, format_str, TZ)
%Convert a string containing a datetime to a datetime

date_pattern = regexprep(format_str, 'yyyy', '\\d{4}');
date_pattern = regexprep(date_pattern, 'yy', '\\d{2}');
date_pattern = regexprep(date_pattern, 'MM', '\\d{2}');
date_pattern = regexprep(date_pattern, 'dd', '\\d{2}');
date_pattern = regexprep(date_pattern, 'HH', '\\d{2}');
date_pattern = regexprep(date_pattern, 'mm', '\\d{2}');
date_pattern = regexprep(date_pattern, 'ss', '\\d{2}');
date_pattern = regexprep(date_pattern, '''T''', '\\T');
date_pattern = regexprep(date_pattern, '''y''', '\y');
date_pattern = regexprep(date_pattern, '''d''', '\d');
date_pattern = regexprep(date_pattern, '''h''', '\h');
date_pattern = regexprep(date_pattern, '''m''', '\m');
date_pattern = regexprep(date_pattern, '''s''', '\s');

[~, datetime_str, ext] = fileparts(str);
match = regexp(datetime_str, date_pattern, 'match');

for i =1:numel(match)
    datetime_out(i,1) = datetime(match{i}, 'InputFormat', format_str, 'Format', 'yyyy-MM-dd''T''HH:mm:ss.SSSZ', 'TimeZone', TZ);
end

end
