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

WELLCOME = 10
RUNNING = 11
GAME_OVER = 12

WHITE = (255,255,255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
GREY = (128, 128, 128)

START_TEXT = 'START'
RETRY_TEXT = 'RETRY'
QUIT_TEXT = 'QUIT'

class SnakeGame:
    def __init__(self, seed = 0,board_szie = 12, silent_mode = True):
        self.board_size = board_szie
        self.grid_size = self.board_size * self.board_size
        self.cell_size = 40
        self.width = self.board_size * self.cell_size
        self.height= self.width

        self.border_size = 20
        self.display_width = self.width + 2 * self.border_size
        self.display_height = self.display_width + 40

        self.silent_mode = silent_mode

        if not self.silent_mode:
            pygame.init()
            pygame.display.set_caption('Snake Game by X')
            self.screen = pygame.display.set_mode((self.display_width,self.display_height))
            self.font = pygame.font.SysFont('arial',25)

            # load sounds
            mixer.init()
            self.eat_sound = mixer.Sound("sounds/eat.wav")
            self.game_over_sound = mixer.Sound("sounds/game_over.wav")
            self.victory_sound = mixer.Sound("sounds/victory.wav")

            head_down = pygame.transform.scale(pygame.image.load("images/snake_head.png"),(50,50))
            head_up = pygame.transform.rotate(head_down,180)
            head_right = pygame.transform.rotate(head_down,90)
            head_left =  pygame.transform.rotate(head_down,-90)
            self.heads = [head_up, head_down, head_left, head_right]
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
        r = self.board_size // 2
        c = r
        self.snake = [(r+1,c),(r,c),(r-1,c)]
        self.direction = DOWN
        self.non_snake = set()
                
        for i in range(self.board_size):
            for j in range(self.board_size):
                if (i,j) not in self.snake:
                    self.non_snake.add((i,j))
        
        self.food = self._gen_food()
        self.score = 0
    
    def _gen_food(self):
        if len(self.non_snake) > 0:
            food = random.sample(list(self.non_snake),1)[0]
        else:
            food = (-1,-1)
        return food
    
    def draw_wellcome(self):
        title = self.font.render('SNAKE GAME',True, WHITE)
        self.screen.fill(BLACK)
        self.screen.blit(title, (self.display_width // 2 - title.get_width() // 2, self.display_height//4))
        start_button = self.font.render(START_TEXT, True, GREY)
        self._draw_button(START_TEXT, (self.display_width // 2, self.display_height // 2))
        self._draw_button(QUIT_TEXT, (self.display_width // 2, self.display_height // 2 + 10 + start_button.get_height()))
        pygame.display.flip()

    def _draw_button(self,text,center_position, hover=WHITE, normal=GREY):
        mouse_pos = pygame.mouse.get_pos()
        button = self.font.render(text,True,normal)
        button_rect = button.get_rect(center = center_position)
        if button_rect.collidepoint(mouse_pos):
            color = self.font.render(text,True,hover)
        else:
            color = self.font.render(text,True,normal)
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
    
    def step(self, action):
        self._update_direction(action)

        # move snake head
        r,c = self.snake[0]
        if self.direction == UP: r -= 1
        elif self.direction == DOWN: r += 1
        elif self.direction == LEFT: c -= 1
        elif self.direction == RIGHT: c += 1

        if (r,c) == self.food:
            food_eaten = True
            self.score += 1
            if not self.silent_mode: self.eat_sound.play()
        else:
            food_eaten = False
            self.non_snake.add(self.snake.pop())

        game_over = (
            (r,c) in self.snake or 
            r < 0 or
            c < 0 or 
            r >= self.board_size or
            c >= self.board_size
        )

        if not game_over:
            self.snake.insert(0,(r,c))
            self.non_snake.remove((r,c))
        else:
            if not self.silent_mode:
                if len(self.snake) <= self.grid_size:
                    self.game_over_sound.play()
                else:
                    self.victory_sound.play()

        if food_eaten:
            self.food = self._gen_food()

        info = {
            'snake_size': len(self.snake),
            'snake_head_position': np.array(self.snake[0]),
            'prev_snake_head_position': np.array(self.snake[1]),
            'food_position': np.array(self.food),
            'food_eated': food_eaten,
            'score': self.score
        }

        return game_over, info
    
    def _update_direction(self, action):
        if action is None:
            return
        if action == UP and self.direction != DOWN: self.direction= UP
        elif action == DOWN and self.direction != UP: self.direction= DOWN
        elif action == LEFT and self.direction != RIGHT: self.direction= LEFT
        elif action == RIGHT and self.direction != LEFT: self.direction= RIGHT

    def draw_countdown(self, number):
        countdown_text = self.font.render(str(number), True, WHITE)
        self.screen.blit(countdown_text, (self.display_width // 2 - countdown_text.get_width() // 2, self.display_height // 2 - countdown_text.get_height() // 2))
        pygame.display.flip() # Update the full display Surface to the screen


    def render(self):
        self.screen.fill(BLACK)

  
        for r in range(self.board_size):
            for c in range(self.board_size):
                if (r%2 == 0 and c%2 == 0) or (r%2 == 1 and c%2 == 1):
                    pygame.draw.rect(self.screen, GREY, (c * self.cell_size+self.border_size,r * self.cell_size+self.border_size,self.cell_size,self.cell_size))
                else:
                    pygame.draw.rect(self.screen, (100,100,100), (c * self.cell_size+self.border_size,r * self.cell_size+self.border_size,self.cell_size,self.cell_size))
        # draw border
        pygame.draw.rect(self.screen, WHITE, (self.border_size - 2, self.border_size - 2, self.width + 4, self.height + 4), 2)

        self.draw_snake()

        self.draw_food()

        self.draw_score()

        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

    def draw_game_over(self):
        game_over_text = self.font.render('GAME OVER', True, RED)
        final_score_text = self.font.render(f'Final Score: {self.score}', True, GREEN)
        self.screen.fill(BLACK)
        self.screen.blit(game_over_text, (self.display_width // 2 - game_over_text.get_width() // 2, self.display_height // 4))
        self.screen.blit(final_score_text, (self.display_width // 2 - final_score_text.get_width() // 2, self.display_height // 4 + 10 + game_over_text.get_height()))
        self._draw_button(RETRY_TEXT, (self.display_width // 2, self.display_height // 2))
        self._draw_button(QUIT_TEXT, (self.display_width // 2, self.display_height // 2 + 10 + start_button.get_height()))
        pygame.display.flip() # Update the full display Surface to the screen

    def draw_score(self):
        score_text = self.font.render(f'Score: {self.score}', True, WHITE)
        self.screen.blit(score_text, (self.board_size, self.height + 2 * self.border_size))

    def draw_food(self):
        if len(self.snake) < self.grid_size:
            r,c = self.food
            pygame.draw.rect(self.screen, RED, (c * self.cell_size+self.border_size,r * self.cell_size+self.border_size,self.cell_size,self.cell_size))

    def draw_snake(self):
        
        # Draw the head
        head_r, head_c = self.snake[0]
        head_x = head_c * self.cell_size + self.border_size
        head_y = head_r * self.cell_size + self.border_size
        """
        # Draw the head (Blue)
        pygame.draw.polygon(self.screen, (100, 100, 255), [
            (head_x + self.cell_size // 2, head_y),
            (head_x + self.cell_size, head_y + self.cell_size // 2),
            (head_x + self.cell_size // 2, head_y + self.cell_size),
            (head_x, head_y + self.cell_size // 2)
        ])

        eye_size = 3
        eye_offset = self.cell_size // 4
        pygame.draw.circle(self.screen, WHITE, (head_x + eye_offset, head_y + eye_offset), eye_size)
        pygame.draw.circle(self.screen, WHITE, (head_x + self.cell_size - eye_offset, head_y + eye_offset), eye_size)
        """
        # Draw the head
        head_img = self.heads[self.direction]
        self.screen.blit(head_img, (head_x-5, head_y-5))

        # Draw the body (color gradient)
        color_list = np.linspace(255, 100, len(self.snake), dtype=np.uint8)
        i = 1
        for r, c in self.snake[1:]:
            body_x = c * self.cell_size + self.border_size
            body_y = r * self.cell_size + self.border_size
            body_width = self.cell_size
            body_height = self.cell_size
            body_radius = 5
            pygame.draw.rect(self.screen, (0, color_list[i], 0),
                            (body_x, body_y, body_width, body_height), border_radius=body_radius)
            i += 1
        # Draw the tail (Pink)
        pygame.draw.rect(self.screen, (255, 100, 100),
                            (body_x, body_y, body_width, body_height), border_radius=body_radius)

if __name__ == '__main__':
    speed = 0
    import time
    seed = random.randint(0,100000000000)
    game = SnakeGame(seed = seed, silent_mode=False)

    game_state = WELLCOME
    print('hello')
    start_button = game.font.render(START_TEXT, True, WHITE)
    retry_button = game.font.render(RETRY_TEXT, True, WHITE)
    quit_button = game.font.render(QUIT_TEXT, True, WHITE)

    action = None
    update_interval = 0.35
    start_time = time.time()

    while True:
        for event in pygame.event.get():
            if game_state == RUNNING:
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_UP: action = UP
                    elif event.key == pygame.K_DOWN: action = DOWN
                    elif event.key == pygame.K_LEFT: action = LEFT
                    elif event.key == pygame.K_RIGHT: action = RIGHT
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if game_state == WELLCOME and event.type == pygame.MOUSEBUTTONDOWN:
                if game.is_mouse_on_button(start_button, position='top'):
                    # count 3 2 1
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
                    # count 3 2 1
                    game.reset()
                    update_interval = 0.35
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
            game.draw_wellcome()
        if game_state == GAME_OVER:
            game.draw_game_over()
        if game_state == RUNNING:
            if time.time() - start_time > update_interval:
                done, info = game.step(action)
                print(info,update_interval)
                update_interval_score = info['score'] // 5
                print(update_interval_score,speed)
                if update_interval_score > speed:
                    speed = update_interval_score
                    update_interval -= 0.01
                game.render()
                start_time = time.time()

                if done: game_state= GAME_OVER
            pygame.time.wait(1)
    