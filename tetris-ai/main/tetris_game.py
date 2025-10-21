import os
import sys
import random
import pygame
import numpy as np
from pygame import mixer

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'

UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3
ROATATE = 4
WELLCOME = 10
RUNNING = 11
GAME_OVER = 12
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
GREY = (100, 100, 100)
START_TEXT = 'START'
RETRY_TEXT = 'RETRY'
QUIT_TEXT = 'QUIT'