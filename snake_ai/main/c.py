import os
import sys
import random
import pygame
import numpy as np
from pygame import mixer

os.environ['PYGMAE_HIDE_SUPPORT_PROMPT'] = '1'

UP = 0
DOWN = 1
LEFT = 3
RIGHT = 4

WELLCOME = 10
RUNNING = 11
GAME_OVER = 12

WHITE = (255,255,255)
BLACK = (0,0,0)
RED = (255,0,0)
GREEN = (0,255,0)
BLUE = (0,0,255)
GREY = (100,100,100)

START_TEXT = '开始'
RETRY_TEXT = '重新开始'
QUIT_TEXT = '离开'

class SnakeGame:
    def __init__(self, seed=0, board_size = 12, silent_mode = True):
        self.seed = seed
        self.board_size = board_size
        self.silent_mode = silent_mode

if __name__ == "__main__":
    import time
    seed = random.randint(0,1000000000)
    game = SnakeGame(seed=seed, silent_mode=False)

    game_state = WELLCOME

    start_button = game.font.render(START_TEXT, True, WHITE)
    retry_button = game.font.render(RETRY_TEXT, True, WHITE)
    quit_button = game.font.render(QUIT_TEXT, True, WHITE)

    action = None 
    update_interval = 0.5
    start_time = time.time()

    while True:
        for event in pygame.event.get():
            if game_state == RUNNING:
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_UP: action = UP
                    elif event.key == pygame.K_DOWN: action = DOWN
                    elif event.key == pygame.K_LEFT:  action = LEFT
                    elif event.key == pygame.K_RIGHT: action = RIGHT
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if game_state == WELLCOME and event.type == pygame.MOUSEBUTTONDOWN:
                if game.is_mouse_on_button(start_button, position='top'):
                    for i in range(3,0,-1):
                        game.screen.fill(BLACK)
                        game.draw_countdown(i)
                        game.eat_sound.play()
                        pygame.time.wait(1000)
                    action = None
                    game_state = RUNNING
                elif game.is_mouse_on_button(quit_button, position='bottom'):
                    pygame.quit()
                    sys.exit()
            if game_state == GAME_OVER and event.type == pygame.MOUSEBUTTONDOWN:
                if game.is_mouse_on_button(retry_button, position='top'):
                    game.reset()
                    for i in range(3,0,-1):
                        game.screen.fill(BLACK)
                        game.draw_countdown(i)
                        game.eat_sound.play()
                        pygame.time.wait(1000)
                    action = None
                    game_state = RUNNING
                elif game.is_mouse_on_button(quit_button, position='bottom'):
                    pygame.quit()
                    sys.exit()
        if game_state == WELLCOME:
            game.draw_welcome()
        if game_state == GAME_OVER:
            game.draw_game_over()
        if game_state == RUNNING:
            if time.time() - start_time > update_interval:
                done,_ = game.step(action)
                game.render()
                start_time = time.time()
                if done: game_state = GAME_OVER
            pygame.time.wait(1)