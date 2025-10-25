import os
import sys
import random
import pygame
import numpy as np
from pygame import mixer

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'

ROTATE = 0
DOWN = 1
LEFT = 2
RIGHT = 3
WELLCOME = 10
RUNNING = 11
GAME_OVER = 12
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
GREY = (100, 100, 100)
LIGHT_GREY = (180, 180, 180)
ORANGE = (255, 165, 0)
VIOLET = (138, 43, 226)
YELLOW = (255, 255, 0)
PINK = (255, 192, 203)
START_TEXT = 'START'
RETRY_TEXT = 'RETRY'
QUIT_TEXT = 'QUIT'

# Base Tetrix class
class Tetrix:
    def __init__(self, board_w=10, board_h=20):
        self.x = board_w // 2 - 2
        self.y = 0
        self.rotation = 0
        self.shape = []
        self.color = WHITE
        self.board_w = board_w
        self.board_h = board_h

    def get_cells(self):
        """Get the current cells occupied by this piece"""
        cells = []
        shape = self.shape[self.rotation]
        for i, row in enumerate(shape):
            for j, cell in enumerate(row):
                if cell:
                    cells.append((self.x + j, self.y + i))
        return cells

    def rotate(self):
        """Rotate the piece clockwise"""
        self.rotation = (self.rotation + 1) % len(self.shape)

    def unrotate(self):
        """Undo rotation"""
        self.rotation = (self.rotation - 1) % len(self.shape)

    def move(self, dx, dy):
        """Move the piece by dx, dy"""
        self.x += dx
        self.y += dy

    def unmove(self, dx, dy):
        """Undo move"""
        self.x -= dx
        self.y -= dy

# I piece - Blue (4 cells in a line)
class Tetrix_I(Tetrix):
    def __init__(self, board_w=10, board_h=20):
        super().__init__(board_w, board_h)
        self.color = BLUE
        self.shape = [
            [[0, 0, 0, 0],
             [1, 1, 1, 1],
             [0, 0, 0, 0],
             [0, 0, 0, 0]],
            [[0, 0, 1, 0],
             [0, 0, 1, 0],
             [0, 0, 1, 0],
             [0, 0, 1, 0]]
        ]

# O piece - Orange (2x2 square)
class Tetrix_O(Tetrix):
    def __init__(self, board_w=10, board_h=20):
        super().__init__(board_w, board_h)
        self.color = ORANGE
        self.shape = [
            [[1, 1],
             [1, 1]]
        ]

# T piece - Green (T shape)
class Tetrix_T(Tetrix):
    def __init__(self, board_w=10, board_h=20):
        super().__init__(board_w, board_h)
        self.color = GREEN
        self.shape = [
            [[0, 1, 0],
             [1, 1, 1],
             [0, 0, 0]],
            [[0, 1, 0],
             [0, 1, 1],
             [0, 1, 0]],
            [[0, 0, 0],
             [1, 1, 1],
             [0, 1, 0]],
            [[0, 1, 0],
             [1, 1, 0],
             [0, 1, 0]]
        ]

# S piece - Violet
class Tetrix_S(Tetrix):
    def __init__(self, board_w=10, board_h=20):
        super().__init__(board_w, board_h)
        self.color = VIOLET
        self.shape = [
            [[0, 1, 1],
             [1, 1, 0],
             [0, 0, 0]],
            [[0, 1, 0],
             [0, 1, 1],
             [0, 0, 1]]
        ]

# Z piece - Red
class Tetrix_Z(Tetrix):
    def __init__(self, board_w=10, board_h=20):
        super().__init__(board_w, board_h)
        self.color = RED
        self.shape = [
            [[1, 1, 0],
             [0, 1, 1],
             [0, 0, 0]],
            [[0, 0, 1],
             [0, 1, 1],
             [0, 1, 0]]
        ]

# J piece - Yellow
class Tetrix_J(Tetrix):
    def __init__(self, board_w=10, board_h=20):
        super().__init__(board_w, board_h)
        self.color = YELLOW
        self.shape = [
            [[1, 0, 0],
             [1, 1, 1],
             [0, 0, 0]],
            [[0, 1, 1],
             [0, 1, 0],
             [0, 1, 0]],
            [[0, 0, 0],
             [1, 1, 1],
             [0, 0, 1]],
            [[0, 1, 0],
             [0, 1, 0],
             [1, 1, 0]]
        ]

# L piece - Pink
class Tetrix_L(Tetrix):
    def __init__(self, board_w=10, board_h=20):
        super().__init__(board_w, board_h)
        self.color = PINK
        self.shape = [
            [[0, 0, 1],
             [1, 1, 1],
             [0, 0, 0]],
            [[0, 1, 0],
             [0, 1, 0],
             [0, 1, 1]],
            [[0, 0, 0],
             [1, 1, 1],
             [1, 0, 0]],
            [[1, 1, 0],
             [0, 1, 0],
             [0, 1, 0]]
        ]

class TetrisGame:
    def __init__(self, seed=42, board_h=20, board_w=10, silent_mode=False):
        self.seed = seed
        random.seed(seed)
        self.silent_mode = silent_mode
        self.cell_size = 20
        self.hidden_h = 4
        self.board_h = board_h
        self.board_w = board_w
        self.height = (self.board_h + self.hidden_h) * self.cell_size
        self.board_right = self.hidden_h * self.cell_size
        self.width = self.board_w * self.cell_size + self.board_right

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
        self.tetrix_classes = [Tetrix_I, Tetrix_O, Tetrix_T, Tetrix_S, Tetrix_Z, Tetrix_J, Tetrix_L]

        # Initialize game board (board_h + hidden_h rows, board_w columns)
        self.board = np.zeros((self.board_h + self.hidden_h, self.board_w), dtype=int)
        self.board_colors = [[BLACK for _ in range(self.board_w)] for _ in range(self.board_h + self.hidden_h)]

        # Current and next pieces
        self.reset()

    def reset(self):
        self.score = 0
        self.level = 1
        random.seed(self.seed)
        self.board = np.zeros((self.board_h + self.hidden_h, self.board_w), dtype=int)
        self.board_colors = [[BLACK for _ in range(self.board_w)] for _ in range(self.board_h + self.hidden_h)]
        self.current_tetrix = None
        self.next_tetrix = self._random_next_tetrix()
        self._spawn_new_tetrix()

    def _random_next_tetrix(self):
        """Generate a random tetrix piece"""
        tetrix_class = random.choice(self.tetrix_classes)
        return tetrix_class(self.board_w, self.board_h)

    def _spawn_new_tetrix(self):
        """Spawn a new tetrix piece"""
        self.current_tetrix = self.next_tetrix
        self.next_tetrix = self._random_next_tetrix()

    def _is_valid_position(self, tetrix):
        """Check if the tetrix position is valid (no collision)"""
        cells = tetrix.get_cells()
        for x, y in cells:
            # Check boundaries
            if x < 0 or x >= self.board_w or y >= self.board_h + self.hidden_h:
                return False
            # Check collision with placed pieces
            if y >= 0 and self.board[y][x]:
                return False
        return True

    def _can_move(self, dx, dy):
        """Check if we can move the current tetrix by dx, dy"""
        self.current_tetrix.move(dx, dy)
        valid = self._is_valid_position(self.current_tetrix)
        self.current_tetrix.unmove(dx, dy)
        return valid

    def _can_rotate(self):
        """Check if we can rotate the current tetrix"""
        self.current_tetrix.rotate()
        valid = self._is_valid_position(self.current_tetrix)
        self.current_tetrix.unrotate()
        return valid

    def _place_tetrix(self):
        """Place the current tetrix on the board"""
        cells = self.current_tetrix.get_cells()
        for x, y in cells:
            if y >= 0:
                self.board[y][x] = 1
                self.board_colors[y][x] = LIGHT_GREY

    def _clear_lines(self):
        """Clear completed lines and return number of lines cleared"""
        lines_cleared = 0
        y = self.board_h + self.hidden_h - 1
        while y >= 0:
            if all(self.board[y]):
                # Remove this line
                self.board = np.delete(self.board, y, 0)
                self.board = np.vstack([np.zeros((1, self.board_w), dtype=int), self.board])

                # Remove color line
                del self.board_colors[y]
                self.board_colors.insert(0, [BLACK for _ in range(self.board_w)])

                lines_cleared += 1
            else:
                y -= 1

        # Update score
        if lines_cleared > 0:
            self.score += lines_cleared * 100 * self.level
            if not self.silent_mode:
                self.eliminate_sound.play()

        # Update level (every 6666 points)
        self.level = self.score // 6666 + 1

        return lines_cleared

    def _drop_direct(self):
        """Drop the tetrix directly to the bottom"""
        while self._can_move(0, 1):
            self.current_tetrix.move(0, 1)

    def _is_game_over(self):
        """Check if game is over (pieces in hidden area)"""
        for y in range(self.hidden_h):
            if any(self.board[y]):
                return True
        return False 

    def draw_wellcome(self):
        title = self.font.render('TETRIS GAME', True, WHITE)
        self.screen.fill(BLACK)
        self.screen.blit(title, (self.display_width // 2 - title.get_width() // 2, self.display_height // 4))
        start_button = self.font.render(START_TEXT, True, GREY)
        self._draw_button(START_TEXT, (self.display_width // 2, self.display_height // 2))
        self._draw_button(QUIT_TEXT, (self.display_width // 2, self.display_height // 2 + 10 + start_button.get_height()))
        pygame.display.flip()

    def draw_game_over(self):
        title = self.font.render('GAME OVER', True, WHITE)
        score_text = self.font.render(f'Score: {self.score}', True, WHITE)
        self.screen.fill(BLACK)
        self.screen.blit(title, (self.display_width // 2 - title.get_width() // 2, self.display_height // 4))
        self.screen.blit(score_text, (self.display_width // 2 - score_text.get_width() // 2, self.display_height // 3))
        retry_button = self.font.render(RETRY_TEXT, True, GREY)
        self._draw_button(RETRY_TEXT, (self.display_width // 2, self.display_height // 2))
        self._draw_button(QUIT_TEXT, (self.display_width // 2, self.display_height // 2 + 10 + retry_button.get_height()))
        pygame.display.flip()
    
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
        """Execute one game step based on the action"""
        done = False

        # Handle player action
        if action == LEFT:
            if self._can_move(-1, 0):
                self.current_tetrix.move(-1, 0)
        elif action == RIGHT:
            if self._can_move(1, 0):
                self.current_tetrix.move(1, 0)
        elif action == DOWN:
            # Direct drop
            self._drop_direct()
            self._place_tetrix()
            self._clear_lines()
            self._spawn_new_tetrix()
            if self._is_game_over():
                done = True
                if not self.silent_mode:
                    self.game_over_sound.play()
            return done, self.score
        elif action == ROTATE:
            if self._can_rotate():
                self.current_tetrix.rotate()

        # Auto move down
        if self._can_move(0, 1):
            self.current_tetrix.move(0, 1)
        else:
            # Place piece and spawn new one
            self._place_tetrix()
            self._clear_lines()
            self._spawn_new_tetrix()
            if self._is_game_over():
                done = True
                if not self.silent_mode:
                    self.game_over_sound.play()

        return done, self.score

    def draw_score(self):
        """Draw the score"""
        score_text = self.font.render(f'SCORE', True, WHITE)
        score_value = self.font.render(f'{self.score}', True, WHITE)
        x = self.border_size + self.board_w * self.cell_size + 30
        y = self.border_size + 150
        self.screen.blit(score_text, (x, y))
        self.screen.blit(score_value, (x, y + 30))

    def draw_next_tetris(self):
        """Draw the next tetrix piece preview"""
        next_text = self.font.render('NEXT', True, WHITE)
        x = self.border_size + self.board_w * self.cell_size + 30
        y = self.border_size + 20
        self.screen.blit(next_text, (x, y))

        # Draw the next piece
        preview_x = x
        preview_y = y + 40
        shape = self.next_tetrix.shape[0]
        for i, row in enumerate(shape):
            for j, cell in enumerate(row):
                if cell:
                    pygame.draw.rect(self.screen, self.next_tetrix.color,
                                   (preview_x + j * 20, preview_y + i * 20, 18, 18))

    def draw_level(self):
        """Draw the level"""
        level_text = self.font.render(f'LEVEL', True, WHITE)
        level_value = self.font.render(f'{self.level}', True, WHITE)
        x = self.border_size + self.board_w * self.cell_size + 30
        y = self.border_size + 250
        self.screen.blit(level_text, (x, y))
        self.screen.blit(level_value, (x, y + 30))

    def render(self):
        """Render the game"""
        if self.silent_mode:
            return

        self.screen.fill(BLACK)

        # Draw the board border
        board_x = self.border_size
        board_y = self.border_size
        pygame.draw.rect(self.screen, WHITE,
                        (board_x - 2, board_y - 2,
                         self.board_w * self.cell_size + 4,
                         self.board_h * self.cell_size + 4), 2)

        # Draw placed pieces (only visible area)
        for y in range(self.hidden_h, self.board_h + self.hidden_h):
            for x in range(self.board_w):
                if self.board[y][x]:
                    pygame.draw.rect(self.screen, self.board_colors[y][x],
                                   (board_x + x * self.cell_size,
                                    board_y + (y - self.hidden_h) * self.cell_size,
                                    self.cell_size - 2, self.cell_size - 2))

        # Draw current tetrix
        if self.current_tetrix:
            cells = self.current_tetrix.get_cells()
            for x, y in cells:
                if y >= self.hidden_h:  # Only draw visible part
                    pygame.draw.rect(self.screen, self.current_tetrix.color,
                                   (board_x + x * self.cell_size,
                                    board_y + (y - self.hidden_h) * self.cell_size,
                                    self.cell_size - 2, self.cell_size - 2))

        # Draw grey bottom area
        grey_y = board_y + self.board_h * self.cell_size + 10
        pygame.draw.rect(self.screen, GREY,
                        (board_x, grey_y, self.board_w * self.cell_size, 20))

        # Draw UI elements
        self.draw_next_tetris()
        self.draw_score()
        self.draw_level()

        pygame.display.flip()
    
if __name__ == "__main__":
    import time
    seed = random.randint(0, 1000000000)
    game = TetrisGame(seed=seed, silent_mode=False)

    game_state = WELLCOME

    start_button = game.font.render(START_TEXT, True, WHITE)
    retry_button = game.font.render(RETRY_TEXT, True, WHITE)
    quit_button = game.font.render(QUIT_TEXT, True, WHITE)

    action = None
    start_time = time.time()

    while True:
        for event in pygame.event.get():
            if game_state == RUNNING:
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_LEFT:
                        action = LEFT
                    elif event.key == pygame.K_RIGHT:
                        action = RIGHT
                    elif event.key == pygame.K_DOWN:
                        action = DOWN
                    elif event.key == pygame.K_UP or event.key == pygame.K_SPACE:
                        action = ROTATE
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if game_state == WELLCOME and event.type == pygame.MOUSEBUTTONDOWN:
                if game.is_mouse_on_button(start_button, position='top'):
                    # count 3 2 1
                    for i in range(3, 0, -1):
                        game.screen.fill(BLACK)
                        game.draw_countdown(i)
                        game.eliminate_sound.play()
                        pygame.time.wait(1000)
                    action = None
                    game_state = RUNNING
                    start_time = time.time()
                elif game.is_mouse_on_button(quit_button, position='bottom'):
                    pygame.quit()
                    sys.exit()
            if game_state == GAME_OVER and event.type == pygame.MOUSEBUTTONDOWN:
                if game.is_mouse_on_button(retry_button, position='top'):
                    game.reset()
                    for i in range(3, 0, -1):
                        game.screen.fill(BLACK)
                        game.draw_countdown(i)
                        game.eliminate_sound.play()
                        pygame.time.wait(1000)
                    action = None
                    game_state = RUNNING
                    start_time = time.time()
                elif game.is_mouse_on_button(quit_button, position='bottom'):
                    pygame.quit()
                    sys.exit()
        if game_state == WELLCOME:
            game.draw_wellcome()
        if game_state == GAME_OVER:
            game.draw_game_over()
        if game_state == RUNNING:
            # Calculate speed based on level (faster as level increases)
            update_interval = max(0.1, 0.5 - (game.level - 1) * 0.05)

            if time.time() - start_time >= update_interval:
                done, _ = game.step(action)
                game.render()
                start_time = time.time()
                action = None  # Reset action after processing

                if done:
                    game_state = GAME_OVER
            pygame.time.wait(1)
    # End of the game