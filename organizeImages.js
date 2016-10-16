/*
Requeriments:
- Node (https://nodejs.org).
- Packages: fs, fs.extra, path.

Instructions:
    Put this in a directory with 'PH2 Dataset images' and execute it.
    e.g.:
    .
    |____index.js
    |____PH2 Dataset images
    | |____IMD002
    | | |____IMD002_Dermoscopic_Image
    | | | |____IMD002.bmp               <------ We need this
    | | |____IMD002_lesion
    | | | |____IMD002_lesion.bmp
    | | |____IMD002_roi
    | | | |____IMD002_R1_Label4.bmp
    | | | |____IMD002_R2_Label3.bmp
    | |____IMD003
    | | |____IMD003_Dermoscopic_Image
    | | | |____IMD003.bmp               <------ We need this
    | | |____IMD003_lesion
    | | | |____IMD003_lesion.bmp
    | | |____IMD003_roi
    | | | |____IMD003_R1_Label4.bmp
    | |____IMD004
    | | |____IMD004_Dermoscopic_Image
    | | | |____IMD004.bmp               <------ We need this
    | | |____IMD004_lesion
    | | | |____IMD004_lesion.bmp
    | | |____IMD004_roi
    | | | |____IMD004_R1_Label4.bmp
    | | | |____IMD004_R2_Label3.bmp
    | |____IMD006
    | | |____...
    | |____IMD008
    | | |____...
    | |____IMD009
    | | |____...
    | |____IMD010
    ...

This algorithm arranges the files, sorting out the ones we need and putting them together in a directory.
e.g.:
.
|____IMD002.bmp
|____IMD003.bmp
|____IMD004.bmp
|____IMD006.bmp
|____IMD008.bmp
|____IMD009.bmp
|____IMD010.bmp
...

*/

'use strict';

const fs = require('fs')
    , fsExtra = require('fs.extra')
    , path = require('path')
    , dest = 'imgs'
    , src = 'PH2 Dataset images';

// Creates the directory if it doesn't exist
if (!fs.existsSync(dest)){
    fs.mkdirSync(dest);
};

fs.readdir(path.normalize(src), (err, files) => {
    // Iterate through files
    files.forEach(file => {
        fs.stat(path.normalize(src + '/' + file), (err, stats) => {
            if (err) {
                console.error(err);
                return;
            };
            if (stats.isDirectory()) {
                // Copy the images to destination folder
                // From 'PH2 Dataset images/IMD###/IMD###_Dermoscopic_Image/IMD###.bmp' to 'imgs/IMD###.bmp'
                fsExtra.copy(path.normalize(src + '/' + file + '/' + file + '_Dermoscopic_Image' + '/' + file + '.bmp'), path.normalize(dest + '/' + file + '.bmp'), { replace: false }, function (err) {
                    if (err) {
                        console.error(err);
                        return;
                    };
                });
            };
        });
    });
});
