#import test

#test.sliceData('../constain6', 'samples')
fileName = 'folder1/folder2/0000_01_02.tif'

subparts = fileName.split('/')
sunpart = subparts[len(subparts) -1]
subparts2 = sunpart.split('.')[0].split('_')
tempName = subparts2[0] + '_' + subparts2[2] + '.tif';


print(tempName)


