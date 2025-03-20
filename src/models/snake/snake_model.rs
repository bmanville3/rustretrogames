use std::{collections::VecDeque, time::Instant};

use log::{debug, warn};
use rand::seq::SliceRandom;

/// Amount of time before snake is forced to move.
pub const MILLIS_BETWEEN_FRAMES: u64 = 300;
/// Cap of how fast a player can make consecutive moves.
pub const PLAYER_MOVEMENT_CAP: u64 = MILLIS_BETWEEN_FRAMES / 10;

#[derive(Clone, Debug)]
pub enum SnakeBlock {
    EMPTY,
    APPLE,
    PLAYERONE,
    HEADONE,
    PLAYERTWO,
    HEADTWO,
}

#[derive(Debug)]
pub struct Snake {
    grid: Vec<Vec<SnakeBlock>>,
    snake_one: VecDeque<(usize, usize)>,
    so_dir: (i8, i8),
    so_last_move: Instant,
    snake_two: VecDeque<(usize, usize)>,
    st_dir: (i8, i8),
    st_last_move: Instant,
    size: u8,
    winner: u8,
}

impl Snake {
    /// # Panics
    ///
    /// Will panic if size does not fit in '[u8]'.
    #[must_use]
    pub fn new() -> Self {
        let size = 32;
        let mut grid: Vec<Vec<SnakeBlock>> = Vec::with_capacity(size);
        for _ in 0..size {
            let mut row: Vec<SnakeBlock> = Vec::with_capacity(size);
            for _ in 0..size {
                row.push(SnakeBlock::EMPTY);
            }
            grid.push(row);
        }
        let mut snake_one = VecDeque::new();
        let player_one_loc = size / 4;
        grid[player_one_loc + 3][player_one_loc + 3] = SnakeBlock::APPLE;
        grid[player_one_loc][player_one_loc] = SnakeBlock::HEADONE;

        let mut snake_two = VecDeque::new();
        let player_two_loc = size * 3 / 4;
        grid[player_two_loc - 3][player_two_loc - 3] = SnakeBlock::APPLE;
        grid[player_two_loc][player_two_loc] = SnakeBlock::HEADTWO;

        snake_one.push_front((player_one_loc, player_one_loc));
        snake_two.push_front((player_two_loc, player_two_loc));

        Self {
            grid,
            snake_one,
            so_dir: (1, 0),
            so_last_move: Instant::now(),
            snake_two,
            st_dir: (-1, 0),
            st_last_move: Instant::now(),
            size: size.try_into().unwrap(),
            winner: 0,
        }
    }

    fn put_random_apple(&mut self) {
        let mut avaliable = Vec::new();
        for i in 0..self.grid.len() {
            for j in 0..self.grid[i].len() {
                if matches!(self.grid[i][j], SnakeBlock::EMPTY) {
                    avaliable.push((i, j));
                }
            }
        }
        if avaliable.is_empty() {
            warn!("No where left to place apples");
            return;
        }
        let index = avaliable.choose(&mut rand::thread_rng());
        if let Some(index) = index {
            self.grid[index.0][index.1] = SnakeBlock::APPLE;
        } else {
            warn!("Unable to get a random index. Choosing middle one found");
            let index = avaliable[avaliable.len() / 2 + 1];
            self.grid[index.0][index.1] = SnakeBlock::APPLE;
        }
    }

    /// # Panics
    ///
    /// Panics if the player's snake is empty or type conversions fail.
    pub fn move_character(
        &mut self,
        delta: (i8, i8),
        character: u8,
        by_timer: bool,
        time: Instant,
    ) {
        if self.winner != 0 {
            return;
        }
        let mut delta = delta;
        let opposition;
        let snake;
        let head_type;
        let block_type;
        if character == 1 {
            // cant do an insta turn around
            if delta == self.so_dir || (-delta.0, -delta.1) == self.so_dir {
                return;
            }
            opposition = 2;
            snake = &mut self.snake_one;
            block_type = SnakeBlock::PLAYERONE;
            head_type = SnakeBlock::HEADONE;
            if by_timer {
                delta = self.so_dir;
            } else {
                self.so_dir = delta;
                self.so_last_move = time;
            }
        } else {
            // cant do an insta turn around
            if delta == self.st_dir || (-delta.0, -delta.1) == self.st_dir {
                return;
            }
            opposition = 1;
            snake = &mut self.snake_two;
            block_type = SnakeBlock::PLAYERTWO;
            head_type = SnakeBlock::HEADTWO;
            if by_timer {
                delta = self.st_dir;
            } else {
                self.st_dir = delta;
                self.st_last_move = time;
            }
        }

        let front = snake.front().unwrap();
        // the grid and snake is bound by u8 so i64 should always cover what is needed
        let new_x: i64 = i64::try_from(front.0).unwrap() + i64::from(delta.0);
        let new_y: i64 = i64::try_from(front.1).unwrap() + i64::from(delta.1);
        let size = i64::from(self.size);
        let mut winner = 0;
        if new_x < size && new_y < size && new_x >= 0 && new_y >= 0 {
            let new_x = usize::try_from(new_x).unwrap();
            let new_y = usize::try_from(new_y).unwrap();
            match self.grid.get(new_x).unwrap().get(new_y).unwrap() {
                SnakeBlock::APPLE => {
                    self.grid[new_x][new_y] = head_type;
                    self.grid[front.0][front.1] = block_type;
                    snake.push_front((new_x, new_y));
                    self.put_random_apple();
                }
                SnakeBlock::EMPTY => {
                    self.grid[new_x][new_y] = head_type;
                    self.grid[front.0][front.1] = block_type;
                    snake.push_front((new_x, new_y));
                    if let Some(old_tail) = snake.pop_back() {
                        self.grid[old_tail.0][old_tail.1] = SnakeBlock::EMPTY;
                    } else {
                        debug!("Removed from back but got none");
                    }
                }
                _ => {
                    winner = opposition;
                }
            }
        } else {
            winner = opposition;
        }
        self.winner = winner;
    }

    #[must_use]
    pub fn get_winner(&self) -> u8 {
        self.winner
    }

    #[must_use]
    pub fn get_grid(&self) -> &Vec<Vec<SnakeBlock>> {
        &self.grid
    }

    #[must_use]
    pub fn get_size(&self) -> u8 {
        self.size
    }

    #[must_use]
    pub fn get_second_player_last_move_time(&self) -> &Instant {
        &self.st_last_move
    }

    #[must_use]
    pub fn get_second_player_direction(&self) -> (i8, i8) {
        self.st_dir
    }

    #[must_use]
    pub fn get_second_player_snae(&self) -> &VecDeque<(usize, usize)> {
        &self.snake_two
    }

    #[must_use]
    pub fn get_first_player_last_move_time(&self) -> &Instant {
        &self.so_last_move
    }

    #[must_use]
    pub fn get_first_player_direction(&self) -> (i8, i8) {
        self.so_dir
    }

    #[must_use]
    pub fn get_first_player_snae(&self) -> &VecDeque<(usize, usize)> {
        &self.snake_one
    }
}

impl Default for Snake {
    fn default() -> Self {
        Self::new()
    }
}
