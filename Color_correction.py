__author__ = 'Mohammad'

import numpy as np
import cv2
import colorbalance
import os

def Color_correct_and_write(card,image,study_file_name, Acc):
    
    card_damaged = False
    card_rotated = False
    correction_error = 0
    
    CardRGB = cv2.cvtColor(card, cv2.COLOR_BGR2RGB)
    ImageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    actual_colors, actual_colors_std = colorbalance.get_colorcard_colors(CardRGB,grid_size=[6, 4])
    
    if any(actual_colors_std>90):
        card_damaged = True
        return card_damaged, card_rotated, correction_error
        # we can comment the above return and use the following two lines if we want to color correct regardless of the corrupted colors
        # actual_colors = np.delete(actual_colors,np.where(actual_colors_std>90),1)
        # true_colors = np.delete(true_colors,np.where(actual_colors_std>90),1)

    
    cnt_color = 0
    # comparing yellow and light red, yellow should have larger value
    if np.sum(actual_colors[:, 8])> np.sum(actual_colors[:, -9]): 
        cnt_color = cnt_color + 1
    # comparing white and blue-green, white should have larger value
    if np.sum(actual_colors[:, 5])> np.sum(actual_colors[:, -6]): 
        cnt_color = cnt_color + 1
    # comparing black and dark tone, black should have smaller value
    if np.sum(actual_colors[:, 0])< np.sum(actual_colors[:, -1]): 
        cnt_color = cnt_color + 1
    # If two or more of the above conditions are met, card is then rotated
    if cnt_color >= 2:
        actual_colors = actual_colors[:, ::-1]
        actual_colors_std = actual_colors_std[::-1]
        card_rotated = True
    
    true_colors = colorbalance.ColorCheckerRGB_CameraTrax
   
  
    iter = 0
    actual_colors2 = actual_colors
    Check = True
    while Check:
        iter = iter + 1
        color_alpha, color_constant, color_gamma = colorbalance.get_color_correction_parameters(true_colors,actual_colors2,'gamma_correction')
        corrected_colors = colorbalance._gamma_correction_model(actual_colors2, color_alpha, color_constant, color_gamma)
        diff_colors = true_colors - corrected_colors
        errors = np.sqrt(np.sum(diff_colors * diff_colors, axis=0)).tolist()
        # Sometimes, although card detection is OK (Acc is high), optimization for
        # color corection fails (high error). In this case, actual_colors are changed
        # slightly an dcorrection is repeated 
        if Acc > 0.4 and np.mean(errors) > 40 and iter < 6:
            actual_colors2 = actual_colors + np.random.rand(3,24)
            # print('   Corrction error high, correcting again....!')
        else:
            Check = False
   
    correction_error = round((np.mean(errors)/255)*10000)/float(100)

    if correction_error < 50:  # equivalent to 20% error
        ImageRGBCorrected = colorbalance.correct_color(ImageRGB, color_alpha,color_constant, color_gamma)
        # get back to RBG order for OpenCV
        ImageCorrected = cv2.cvtColor(ImageRGBCorrected, cv2.COLOR_RGB2BGR)
        if not os.path.exists(os.path.dirname(study_file_name)):
            os.makedirs(os.path.dirname(study_file_name))
        cv2.imwrite(study_file_name,ImageCorrected)
    
    return card_damaged, card_rotated, correction_error

