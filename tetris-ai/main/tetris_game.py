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
ROTATE = 4
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

class TetrisGame:
    def __init__(self, seed= 42, board_h=20, board_w=10, silent_mode=False):
        self.cell_size = 30
        self.hidden_h = 4
        self.board_h = board_h
        self.board_w = board_w
        self.height = (self.board_h + self.hidden_h) * self.cell_size
        self.board_right = self.hidden_h * self.cell_size
        self.width = (self.board_w + self.board_right )* self.cell_size

        self.border_size = 20
        self.display_height = self.height + self.border_size * 2 + 40
        self.display_width = self.width + self.border_size * 3 + 40
        
        if not self.silent_mode:
            pygame.init()
            pygame.display.set_caption('Tetris By ilun')
            self.screen = pygame.display.set_mode((self.display_width, self.display_height))
            self.font = pygame.font.SysFont('arial', 25)

            # Load sounds
            mixer.init()
            self.eliminate_sound = mixer.Sound("sounds/eliminate.wav")
            self.game_over_sound = mixer.Sound("sounds/game_over.wav")
            self.victory_sound = mixer.Sound("sounds/victory.wav")
        else:
            self.screen = None
            self.font = None

        self.score = 0
        self.level = 1

    def reset(self):
        self.score = 0
        self.level = 1

    def _random_next_tetris(self):
        return 

    
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

    def is_mouse_on_button(self, button, position='top'):
        """
        Check if mouse is on a specific button
        Args:
            button: pygame rendered text surface
            position: 'top' for start/retry button, 'bottom' for quit button
        """
        mouse_pos = pygame.mouse.get_pos()
        if position == 'top':
            button_rect = button.get_rect(center=(self.display_width // 2, self.display_height // 2))
        else:  # bottom position for quit button
            button_rect = button.get_rect(center=(self.display_width // 2, self.display_height // 2 + 10 + button.get_height()))
        
        return button_rect.collidepoint(mouse_pos)
    
    def draw_countdown(self, number):
        countdown_text = self.font.render(str(number), True, WHITE)
        self.screen.blit(countdown_text, (self.display_width // 2 - countdown_text.get_width() // 2, self.display_height // 2 - countdown_text.get_height() // 2))
        pygame.display.flip() # Update the full display Surface to the screen

    def step(self, action):
        print("TODO")
        return 
    
    def draw_score(self):
        print("TODO")
        return
    
    def draw_next_tetris(self):
        print("TODO")
        return
    
    def draw_level(self):
        print("TODO")
        return
    
    def render(self):
        print("TODO")
        return
    
if __name__ == "__main__":
    import time
    game = TetrisGame(silent_mode=False)
    seed = random.randint(0, 1000000000)
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
                    print("TODO")
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if game_state == WELLCOME and event.type == pygame.MOUSEBUTTONDOWN:
                if game.is_mouse_on_button(start_button, position='top'):
                    # count 3 2 1
                    for i in range(3, 0, -1):
                        game.screen.fill(BLACK)
                        game.draw_countdown(i)
                        game.eliminate.play()
                        pygame.time.wait(1000)
                    action = None
                    game_state = RUNNING
                elif game.is_mouse_on_button(quit_button, position='bottom'):
                    pygame.quit()
                    sys.exit()
            if game_state == GAME_OVER and event.type == pygame.MOUSEBUTTONDOWN:
                if game.is_mouse_on_button(retry_button, position='top'):
                    game.reset()
                    for i in range(3, 0, -1):
                        game.screen.fill(BLACK)
                        game.draw_countdown(i)
                        game.eliminate.play()
                        pygame.time.wait(1000)
                    action = None
                    game_state = RUNNING
                elif game.is_mouse_on_button(quit_button, position='bottom'):
                    pygame.quit()
                    sys.exit()
        if game_state == WELLCOME:
            game.draw_wellcome()
        if game_state == GAME_OVER:
            game.draw_game_over()
        if game_state == RUNNING:
            if time.time() - start_time >= update_interval:
                done, _ = game.step(action)
                game.render()
                start_time = time.time()

                if done: game_state = GAME_OVER
            pygame.time.wait(1)
    # End of the game