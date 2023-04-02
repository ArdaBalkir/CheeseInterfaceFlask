import tensorflow as tf
import numpy as np
from PIL import Image
from custom_methods import *

def report_cheddar(image_array):

    sat, bright = get_saturation_and_brightness(image_array)
    return f'The cheese type is Cheddar!, The cheese has a brightness of {bright} out of 255 and saturation of {sat} out of 255'

def report_emmantal(image_array):

    hole_count, hole_to_surface_ratio = holiness(image_array)
    sat, bright = get_saturation_and_brightness(image_array)
    return f'The cheese type is Emmantal!, with {hole_count} holes and {hole_to_surface_ratio} hole to surface ratio!\nThe cheese has a brightness of {bright} out of 255 and saturation of {sat} out of 255'

def report_leicester(image_array):

    sat, bright = get_saturation_and_brightness(image_array)
    return f'The cheese type is Leicester!, The cheese has a brightness of {bright} out of 255 and saturation of {sat} out of 255'
