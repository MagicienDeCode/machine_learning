import os
import sys
import random
import pygame
import numpy as np
from pygame import mixer

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'

UP = 1
DOWN = 2
LEFT = 3
RIGHT = 4
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

class SnakeGame:
    def __init__(self, seed = 0, board_size = 12, silent_mode = True):
        self.board_size = board_size
        self.grid_size = self.board_size * self.board_size # self.grid_size = self.board_size ** 2 
        self.cell_size = 40
        self.width = self.board_size * self.cell_size
        self.height = self.width

        self.border_size = 20
        self.display_width = self.width + 2 * self.border_size
        self.display_height = self.display_width + 40

        self.silent_mode = silent_mode
        if not self.silent_mode:
            os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '0'
            pygame.init()
            pygame.display.set_caption('Snake Game By ilun')
            self.screen = pygame.display.set_mode((self.display_width, self.display_height))
            self.font = pygame.font.SysFont('arial', 25)

            # Load sounds
            mixer.init()
            self.eat_sound = mixer.Sound("sounds/eat.wav")
            self.game_over_sound = mixer.Sound("sounds/game_over.wav")
            self.victory_sound = mixer.Sound("sounds/victory.wav")
        else:
            self.screen = None
            self.font = None
        
        self.snake = None
        self.non_snake = None
        self.direction = None
        self.food = None
        self.seed = seed

        random.seed(self.seed)

        self.reset()
    
    def reset(self):
        x = self.board_size // 2
        y = self.board_size // 2
        self.snake = [(x+1, y),(x, y),(x-1, y)]
        self.non_snake = set()
        # self.non_snake = set([(r,c) for r in range(self.board_size) for c in range(self.board_size) if (r,c) not in self.snake])
        for r in range(self.board_size):
            for c in range(self.board_size):
                if (r,c) not in self.snake:
                    self.non_snake.add((r,c))

        self.direction = DOWN
        self.food = self._gen_food()
        self.score = 0
    
    def _gen_food(self):
        if len(self.non_snake) > 0:
            food = random.sample(list(self.non_snake), 1)[0]
        else:
            food = (-1, -1)
        return food
    
    def draw_wellcome(self):
        title = self.font.render('SNAKE GAME', True, WHITE)
        self.screen.fill(BLACK)
        self.screen.blit(title, (self.display_width // 2 - title.get_width() // 2, self.display_height // 4))
        start_button = self.font.render(START_TEXT, True, GREY)
        self._draw_button(START_TEXT, (self.display_width // 2, self.display_height // 2))
        self._draw_button(QUIT_TEXT, (self.display_width // 2, self.display_height // 2 + 10 + start_button.get_height()))
        pygame.display.flip() # Update the full display Surface to the screen

    def _draw_button(self, text, center_postion, hover=WHITE, normal=GREY):
        mouse_pos = pygame.mouse.get_pos()
        button = self.font.render(text, True, normal)
        button_rect = button.get_rect(center=center_postion)
        if button_rect.collidepoint(mouse_pos):
            color = self.font.render(text, True, hover)
        else:
            color = self.font.render(text, True, normal)
        self.screen.blit(color, button_rect)

if __name__ == "__main__":
    import time
    seed = random.randint(0, 1000000000)
    game = SnakeGame(seed=seed, silent_mode=False)
    
    game_state = WELLCOME

    start_button = game.font.render(START_TEXT, True, WHITE)
    retry_button = game.font.render(RETRY_TEXT, True, WHITE)
    quit_button = game.font.render(QUIT_TEXT, True, WHITE)

    action = None
    update_interval = 0.15
    start_time = time.time()

    while True:
        for event in pygame.event.get():
            if game_state == RUNNING:
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_UP:
                        action = UP
                    elif event.key == pygame.K_DOWN:
                        action = DOWN
                    elif event.key == pygame.K_LEFT:
                        action = LEFT
                    elif event.key == pygame.K_RIGHT:
                        action = RIGHT
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if game_state == WELLCOME and event.type == pygame.MOUSEBUTTONDOWN:
                if game.is_mouse_on_button(start_button):
                    # count 3 2 1
                    for i in range(3, 0, -1):
                        game.screen.fill(BLACK)
                        game.draw_countdown(i)
                        game.eat_sound.play()
                        pygame.time.wait(1000)
                    action = None
                    game_state = RUNNING
                elif game.is_mouse_on_button(quit_button):
                    pygame.quit()
                    sys.exit()
            if game_state == GAME_OVER and event.type == pygame.MOUSEBUTTONDOWN:
                if game.is_mouse_on_button(retry_button):
                    game.reset()
                    for i in range(3, 0, -1):
                        game.screen.fill(BLACK)
                        game.draw_countdown(i)
                        game.eat_sound.play()
                        pygame.time.wait(1000)
                    action = None
                    game_state = RUNNING
                elif game.is_mouse_on_button(quit_button):
                    pygame.quit()
                    sys.exit()
            if game_state == WELLCOME:
                game.draw_wellcome()
            if game_state == GAME_OVER:
                game.draw_game_over()
            if game_state == RUNNING:
                if time.time() - start_time >= update_interval:
                    done, _ = game.play_step(action)
                    game.render()
                    start_time = time.time()

                    if done: game_state = GAME_OVER
            pygame.time.wait(1)
    # End of the game

