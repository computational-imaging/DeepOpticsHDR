#Chris Metzler 2020

#Take every 10th frame from Boitard, MPI, DML-hDR, Stuttgart, and LiU HDRV datasets

#Create a copy of the HDR_videos directory and then decimate its contents

import os

input_dir='./HDR_videos_decimated'#Should have all the video frames to start


def getListOfEXRFiles(dirName):
    #From https://thispointer.com/python-how-to-get-list-of-files-in-directory-and-sub-directories/
    # create a list of file and sub directories
    # names in the given directory
    listOfFile = os.listdir(dirName)
    allFiles = list()
    # Iterate over all the entries
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(dirName, entry)
        # If entry is a directory then get the list of files in this directory
        if os.path.isdir(fullPath):
            allFiles = allFiles + getListOfEXRFiles(fullPath)
        else:
            if fullPath.endswith(".exr") or fullPath.endswith(".hdr"):
                allFiles.append(fullPath)

    return allFiles

EXR_list=[]#A list of all the EXR files
EXR_list=getListOfEXRFiles(input_dir)
# for root, subdirs, files in os.walk(input_dir):
#     print(files)
#     if files.endswith(".exr"):
#         EXR_list.append(os.path.join(input_dir, file))
#         print(1)

EXR_to_save=EXR_list[0::10]#Save every 10th frame of the videos

EXR_to_delete=[x for x in EXR_list if x not in EXR_to_save]

for file in EXR_to_delete:
    os.remove(file)