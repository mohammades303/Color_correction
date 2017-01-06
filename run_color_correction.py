##############################################################################################
#
#       This is a function developed for color card detection and color correction.
#
#       Mandatory inputs:
#           - a folder containing images 
#           - a card template 
#           - an output folder
#       Optional inputs for:
#           - handling vertical color cards
#           - specifying number of cores
#           - specifying rotation range
#           - specifying scale range
#           - enabling fast mode by giving card coordinates
#           - modifying output file name
#           - writing logs
#
#       Mohammad Esmaeilzadeh, Borevitz lab, Australian National University
#       mohammad.esmaeilzadeh@anu.edu.au, m.esmaielzadeh@gmail.com           
#
##############################################################################################


import numpy as np
import os
import cv2
import matplotlib.pylab as plt
from Detect_colorChecker import detect_card
from docopt import docopt
from Color_correction import Color_correct_and_write
import time
import multiprocessing
import logging


NUM_CORES = multiprocessing.cpu_count()



OPTS = """
USAGE:
    run_color_correction -i INPUT -o OUTPUT -c CARD
    run_color_correction -i INPUT -o OUTPUT -c CARD [-v] [(-t NUM_CORES)] [(-d DEGREE)]  [(-s SCALE)] [(-f CORD)] [(-n NAMES)] [(-l LOG)]
    
    run_color_correction -h | --help
OPTIONS:
    -h --help       Show this screen
    -i INPUT        Path to input folder
    -o OUTPUT       Path to output folder
    -c CARD         Path to colorcard
    -s SCALE        Optional, for scaling of card template; SCALE = [low,high,number]
                    e.g. with "-s [0.95,1.05,11]", eleven different scales in the 
                    mentioned range are tested and best scale is selected 
                    Without "-s", no scaling is performed
    -d DEGREE       Optional, for rotation of card template; DEGREE = [low,high,number]
                    e.g. with "-d [-2,2,9]", nine different degrees for rotation in 
                    the mentioned range are tested and best one is selected
                    Without "-t", default range [-2.5 -2 ... 0 ... 2 2.5] is used
    -f CORD         Optional, for faster card detection; CORD = [x_cord,ycord]
                    Coordinates of a point on the colorcard
                    x_cord: relevant to width, y_cord: relevant to height
    -v              Should be used if colorcard is portrait and image is landscape or vice versa
    -t NUM_CORES    Optional number of cores for parallel processing; 
                    Default value = all available cores
    -n NAMES        Optional, to modify output file name; NAMES = [old,new]
                    e.g. with "-n [-orig,-cor]", if '-orig' exists in 
                    the file name, it is replaced with '-cor'
    -l LOG          Optional, enables writing logs into a file. LOG is the folder
                    where the log is saved
"""

def path_exists(x):
    """Validator for path field."""
    x = x.replace('\\', '/')
    if os.path.exists(x):
        return os.path.join(x, '', '')
    raise ValueError("path '%s' doesn't exist" % x)
    
def path_exists2(x):
    """Validator for path field, without raising error"""
    x = x.replace('\\', '/')
    if os.path.exists(x):
        return os.path.join(x, '', '')
    return False
	
def file_exists(x):
    """Validator for file field."""
    x = x.replace('\\', '/')
    if os.path.isfile(x):
        return x
    else:
        raise ValueError("file '%s' doesn't exist" % x)
        # return (False,x)
        
def string_array_check(str_array,length):
    """Validator for length of input array."""
    array = eval(str_array)
    if len(array) is length:
        return(array)
    else:
        raise ValueError("number of variables in '%s' incorrect" % array)

def check_name_modification(names):
    """Validator for filename modification parameter"""
    if names[0] == '[':
        names[0] = ''
    #else:
        #raise ValueError("incorrect format '%s', [] missing" % "".join(names))
    if names[-1] == ']':
        names[-1] = ''
    #else:
        #raise ValueError("incorrect format '%s', [] missing" % "".join(names))
    names = "".join(names)
    names_corrected = [x.strip() for x in names.split(',')]
    if len(names_corrected) is 2:
        return(names_corrected[0], names_corrected[1])
    else:
        raise ValueError("number of variables in '%s' incorrect" % names)


def parse_options(opts):
    Options = { 'vertical' : False,
                'scale_lower' : 1,
                'scale_upper' : 1,
                'scale_num' : 1,
                'degree_lower' : -2.5,
                'degree_upper' : 2.5,
                'degree_num' : 11,
                'fast' : False,
                'cord_x' : '',
                'cord_y' : '',
                'num_cores' : NUM_CORES,
                'modify_name' : False,
                'old_name' : '',
                'new_name' : '',
                'write_log' : False,
                'log_folder' : ''
              }
    if opts["-v"]:
        Options['vertical'] = True
    if opts["-s"] is not None:
        Scale = string_array_check(opts["-s"],3)
        if Scale[0] > 0 and Scale[1] > 0 and Scale[2] > 0:
            Options['scale_lower'] = Scale[0]
            Options['scale_upper'] = Scale[1]
            Options['scale_num'] = Scale[2]
        else:
            raise ValueError("Positive values in scale parameters '%s' expected" % Scale)
        
    if opts["-d"] is not None:
        Degree = string_array_check(opts["-d"],3)
        Options['degree_lower'] = Degree[0]
        Options['degree_upper'] = Degree[1]
        if Degree[2] > 0:
            Options['degree_num'] = Degree[2]
        else:
            raise ValueError("Number of samples '%s' should be positive" % Degree[2])

    if opts["-f"] is not None:
        Options['fast'] = True
        Cord = string_array_check(opts["-f"],2)
        Options['cord_x'] = Cord[0]
        Options['cord_y'] = Cord[1]
    if opts["-n"] is not None:
        Options['modify_name'] = True
        names = list(opts["-n"])
        Options['old_name'], Options['new_name'] = check_name_modification(names)
    if opts["-t"] is not None:
        Options['num_cores'] = int(opts["-t"])
    if opts["-l"] is not None:
        Options['write_log'] = True
        Options['log_folder'] = opts["-l"]

    return(Options)
        
def crop_image(image_orig,height_card,width_card,x_cord,y_cord):
    x_start = max(0,x_cord - int(width_card*1.25))
    y_start = max(0,y_cord - int(height_card*1.25))
    x_end = min(np.size(image_orig,1)-1,x_cord + int(width_card*1.25))
    y_end = min(np.size(image_orig,0)-1,y_cord + int(height_card*1.25))
    image_cropped = image_orig[y_start:y_end,x_start:x_end,:]
    if np.size(image_cropped,0) < height_card or np.size(image_cropped,1) < width_card:
        print("Colorcard coordinate provided for fast mode incorrect. Trying normal mode")
        return(image_orig)
    return(image_cropped)

    
def handle_verical_horizontal_cards(image_resized,vertical):
    height_image = np.size(image_resized,0)
    width_image = np.size(image_resized,1)
    if (height_image > width_image) & (vertical is False):
        image_resized = np.rot90(image_resized)
    if (height_image < width_image) & (vertical is True):
        image_resized = np.rot90(image_resized)
    return(image_resized)
    
        
def main(input_dir,output_dir,card_path,Options):
    try:
        input_dir = path_exists(input_dir)
        output_dir = path_exists(output_dir)
        card_path = file_exists(card_path)
        scale_search_range = np.linspace(Options['scale_lower'],Options['scale_upper'],Options['scale_num'])
        degree_search_range = np.linspace(Options['degree_lower'],Options['degree_upper'],Options['degree_num'])
    except ValueError as e:
        print ("Error on entry", e)
        return
    
    if Options['write_log']:
        LOG_folder = path_exists2(Options['log_folder'])
        if LOG_folder:
            LOG_FILENAME = LOG_folder + 'Log.log'
            logging.basicConfig(filename=LOG_FILENAME,level=logging.INFO)
        else:
            Options['write_log'] = False
            print("\n\nLog folder '%s' not found! "% Options['log_folder'] )
            print('Writing logs disabled')
                  
            
    card = cv2.imread(card_path,0)
    Default_card_h = 100
    height_card = np.size(card,0)
    width_card = np.size(card,1)
    if height_card > width_card:
        card = np.rot90(card)
        height_card, width_card = width_card, height_card
    scale = Default_card_h/float(height_card)
 
    card = cv2.resize(card,(0,0), fx=scale, fy=scale)
    card = cv2.Canny(card, 40, 50)

    for dirpath, dirnames, filenames in os.walk(input_dir):
        for f in filenames:
            full_output_name = os.path.join(dirpath.replace(input_dir,output_dir),f.replace(Options['old_name'], Options['new_name']))
            base, ext = os.path.splitext(f)
            if ext.lower() in ['.jpg', '.png', '.tif', '.tiff']:
                fp = os.path.join(dirpath, f)
                t = time.time()
                if Options['write_log']:
                    logging.info(' Image %s found!',fp)

                print('\n\nProcessing ' + fp)
                print('-- Reading image...')
                image_orig = cv2.imread(fp)
                if Options['fast']:
                    print('-- Cropping image for faster analysis...')
                    image_cropped = crop_image(image_orig, height_card, width_card, Options['cord_x'], Options['cord_y'])
                else:
                    image_cropped = image_orig
                image_resized = cv2.resize(image_cropped,(0,0), fx=scale, fy=scale)
                image_resized = handle_verical_horizontal_cards(image_resized,Options['vertical'])
                #t = time.time()
                print('-- Detecting Colorcard...')
                Detected_card, Acc, SCALE = detect_card(image_resized,card,scale_search_range, degree_search_range, Options['num_cores'])
                print('   Detection accuracy = ' + str(round(Acc*10000)/float(100)) + ' %')
                if Options['write_log']:
                    logging.info('      Detection accuracy = ' + str(round(Acc*10000)/float(100)) + ' %')
                if Acc > 0.3:
                    card_damaged, card_rotated, correction_error = Color_correct_and_write(Detected_card,image_orig,full_output_name, Acc)
                    if card_rotated:
                        # a detected card is rotated if the black sqaure is not in the lowest row
                        print('   Detected card is rotated')
                        if Options['write_log']:
                            logging.warning('   Detected card is rotated')

                    if card_damaged:
                        print('   Color card seems damaged')
                        print('-- Skipping color correction')
                        if Options['write_log']:
                            logging.warning('   Color card seems damaged')
                            logging.error('     Color correction skipped \n')
                    else:
                        print('-- Correcting colors...')
                        print('   Expected correction error = ' + str(correction_error) + ' %')
                        if Options['write_log']:
                            logging.info('      Correction error = ' + str(correction_error) + ' %')
                        if correction_error < 50: 
                            print('-- Writing corrected image:')
                            print('   ' + full_output_name)
                            if Options['write_log']:
                                logging.info('      Corrected image written: \n                ' + full_output_name + '\n')
                        else:
                            print('   Image correction unsatisfactory!')
                            print('   Writing of corrected image skipped')
                            if Options['write_log']:
                                logging.warning('   Image correction unsatisfactory!')
                                logging.error('     Writing of corrected image skipped \n')
                else:
                    print('   Card detection unsatisfactory')
                    print('-- Skipping color correction')
                    if Options['write_log']:
                        logging.warning('   Card detection unsatisfactory')
                        logging.error('     Color correction skipped \n')

                # print(t - time.time())

        
if __name__ == '__main__':
    opts = docopt(OPTS)
    # print(opts)
    Options = parse_options(opts)
    print('\nOptions:')
    print(Options)
    main(opts["-i"],opts["-o"],opts["-c"],Options)
