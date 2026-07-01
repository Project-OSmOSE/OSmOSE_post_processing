function [files] = uigetdir2(start_path, dialog_title)

% if nargin == 0 || start_path == '' || start_path == 0 % Allow a null argument.
if nargin == 0 || isequal(start_path, '') || isequal(start_path, 0) % Allow a null argument.
    start_path = pwd;
end


import com.mathworks.mwswing.MJFileChooserPerPlatform;
jchooser = javaObjectEDT('com.mathworks.mwswing.MJFileChooserPerPlatform', start_path);
jchooser.setFileSelectionMode(javax.swing.JFileChooser.DIRECTORIES_ONLY);
jchooser.setMultiSelectionEnabled(true);
jchooser.setDialogTitle(dialog_title);

jchooser.showOpenDialog([]);

if jchooser.getState() == javax.swing.JFileChooser.APPROVE_OPTION
    jFiles = jchooser.getSelectedFiles();
    files = arrayfun(@(x) char(x.getPath()), jFiles, 'UniformOutput', false);
elseif jchooser.getState() == javax.swing.JFileChooser.CANCEL_OPTION
    files = [];
else
    error('Error occurred while picking file');
end
