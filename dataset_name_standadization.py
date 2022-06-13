import os
import glob

current_files = glob.glob('./*.jpg')


for i, filename in enumerate(current_files):
    #image_number = i+628
    os.rename(filename, './' + '0'*(4-len(str(i))) + str(i) + '.png')
