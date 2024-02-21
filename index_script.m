% Set the directory where your labels folder is located
labelsDir = 'C:\Users\44739\Downloads\plane2\valid\labels';

% Get a list of all txt files in the labels folder
fileList = dir(fullfile(labelsDir, '*.txt'));

% Loop through each file and update the class indexes
for i = 1:length(fileList)
    filePath = fullfile(labelsDir, fileList(i).name);
    
    % Read the content of the txt file
    fileContent = fileread(filePath);
    
    % Split the content into lines
    lines = strsplit(fileContent, '\n');
    
    % Loop through each line and modify the class indexes
    for j = 1:length(lines)
        line = lines{j};
        
        % Skip empty lines
        if isempty(line)
            continue;
        end
        
        % Split the line into tokens
        tokens = strsplit(line);
        
        % Update class indexes 2 to 4 and 3 to 5
        for k = 1:length(tokens)
            if strcmp(tokens{k}, '0')
                tokens{k} = '1';
            end
        end
        
        % Reconstruct the modified line
        lines{j} = strjoin(tokens, ' ');
    end
    
    % Join all the modified lines back into the file content
    fileContent = strjoin(lines, '\n');
    
    % Write the updated content back to the file
    fid = fopen(filePath, 'w');
    fprintf(fid, '%s', fileContent);
    fclose(fid);
end
