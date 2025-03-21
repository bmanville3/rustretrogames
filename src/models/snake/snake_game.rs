use std::{collections::VecDeque, time::Instant};

use log::{debug, info, warn};
use rand::{seq::SliceRandom, Rng};

use super::snake_player::SnakePlayer;

/// Amount of time before snake is forced to move.
pub const MILLIS_BETWEEN_FRAMES: u64 = 300;
/// Cap of how fast a player can make consecutive moves.
pub const PLAYER_MOVEMENT_CAP: u64 = MILLIS_BETWEEN_FRAMES / 10;

// These are all defined as usize since they are used a lot with indexing stuff.

/// Max number of real players allowed.
pub const MAX_NUM_OF_REAL_PLAYERS: usize = 2;
/// Max num of total players allowed.
pub const MAX_NUM_OF_TOTAL_PLAYERS: usize = 6;
/// Max board size.
pub const MAX_BOARD_SIZE: usize = 40;
/// Min board size.
pub const MIN_BOARD_SIZE: usize = 10;
/// Minimum player spawn distance.
pub const MIN_INCR: usize = 5;
/// Boundary between starting positions and wall required.
pub const BOUNDARY: usize = 4;

type Result<T> = std::result::Result<T, SnakeError>;

#[derive(Debug, Clone)]
pub enum SnakeError {
    InvalidPlayerCount,
    InvalidBoardSize,
    InvalidBoardPlayerRatio,
    NotEnoughStarts,
    TooManyReals,
}

/// Type of block that can be found on the board.
#[derive(Clone, Debug)]
pub enum SnakeBlock {
    Empty,
    Apple,
    SnakeBody(usize),
    SnakeHead(usize),
}

/// Action a [`SnakePlayer`] can take in the [`SnakeGame`].
#[derive(Clone, Debug)]
pub enum SnakeAction {
    Up,
    Down,
    Left,
    Right,
}

impl SnakeAction {
    pub const VARIANTS: &'static [SnakeAction] = &[Self::Up, Self::Down, Self::Left, Self::Right];

    #[must_use]
    pub fn value(&self) -> (i8, i8) {
        match self {
            SnakeAction::Up => (-1, 0),
            SnakeAction::Down => (1, 0),
            SnakeAction::Left => (0, -1),
            SnakeAction::Right => (0, 1),
        }
    }

    #[must_use]
    pub fn get_opposite(&self) -> SnakeAction {
        match self {
            SnakeAction::Up => SnakeAction::Down,
            SnakeAction::Down => SnakeAction::Up,
            SnakeAction::Left => SnakeAction::Right,
            SnakeAction::Right => SnakeAction::Left,
        }
    }

    #[must_use]
    pub fn get_random_action() -> SnakeAction {
        SnakeAction::VARIANTS[rand::thread_rng().gen_range(0..SnakeAction::VARIANTS.len())].clone()
    }
}

/// Model of the Snake Game.
#[derive(Clone, Debug)]
pub struct SnakeGame {
    grid: Vec<Vec<SnakeBlock>>,
    snakes: Vec<SnakePlayer>,
    board_size: usize,
    winner: Option<usize>,
}

impl SnakeGame {
    /// Creates a new snake game with the specified hyperparamters.
    ///
    /// # Errors
    ///
    /// If the combination of hyperparameters is invalid (as specified by the constraints),
    /// a [`SnakeError`] is returned.
    ///
    /// # Panics
    ///
    /// Panics if casting from usize to i64 and back fails. This is never expected to happen.
    pub fn new(num_of_bots: usize, num_of_real_players: usize, board_size: usize) -> Result<Self> {
        let num_players = num_of_bots + num_of_real_players;
        // Error if the game is invalid
        if !(1..=MAX_NUM_OF_TOTAL_PLAYERS).contains(&num_players) {
            return Err(SnakeError::InvalidPlayerCount);
        }
        if !(MIN_BOARD_SIZE..=MAX_BOARD_SIZE).contains(&board_size) {
            return Err(SnakeError::InvalidBoardSize);
        }
        if num_of_real_players > MAX_NUM_OF_REAL_PLAYERS {
            return Err(SnakeError::TooManyReals);
        }
        let incr = (board_size - 2 * BOUNDARY) / num_players;
        if num_players > 1 && incr < MIN_INCR {
            return Err(SnakeError::InvalidBoardPlayerRatio);
        }
        let mut grid: Vec<Vec<SnakeBlock>> = Vec::with_capacity(board_size);
        for _ in 0..board_size {
            let mut row: Vec<SnakeBlock> = Vec::with_capacity(board_size);
            for _ in 0..board_size {
                row.push(SnakeBlock::Empty);
            }
            grid.push(row);
        }

        let mut snakes: Vec<SnakePlayer> = Vec::with_capacity(num_players);

        if num_players == 1 {
            let player_one_loc = board_size / 4;
            grid[player_one_loc][player_one_loc] = SnakeBlock::SnakeHead(0);
            grid[player_one_loc - 1][player_one_loc] = SnakeBlock::SnakeBody(0);
            let mut ns = SnakePlayer::new(
                num_of_bots == 1,
                player_one_loc,
                player_one_loc,
                0,
                SnakeAction::Down,
            );
            ns.squares_taken
                .push_back((player_one_loc - 1, player_one_loc));
            snakes.push(ns);
        } else {
            let mut starting_positions: Vec<(usize, usize)> = Vec::new();
            let mut i = BOUNDARY;
            while i < board_size - BOUNDARY {
                let mut j = BOUNDARY;
                while j < board_size - BOUNDARY {
                    starting_positions.push((i, j));
                    j += incr;
                }
                i += incr;
            }
            if starting_positions.len() < num_players {
                return Err(SnakeError::NotEnoughStarts);
            }
            let mut are_bots = vec![false; num_of_real_players];
            are_bots.extend(vec![true; num_of_bots]);
            for (i, is_bot) in are_bots.iter().enumerate() {
                let start_indx = rand::thread_rng().gen_range(0..starting_positions.len());
                let start = starting_positions[start_indx];
                let starting_action = SnakeAction::get_random_action();
                let op = starting_action.get_opposite().value();
                let mut ns = SnakePlayer::new(*is_bot, start.0, start.1, i, starting_action);
                grid[start.0][start.1] = SnakeBlock::SnakeHead(i);
                // skill issues of adding numbers shines through again
                let bx =
                    usize::try_from(i64::try_from(start.0).unwrap() + i64::from(op.0)).unwrap();
                let by =
                    usize::try_from(i64::try_from(start.1).unwrap() + i64::from(op.1)).unwrap();
                grid[bx][by] = SnakeBlock::SnakeBody(i);
                ns.squares_taken.push_back((bx, by));
                snakes.push(ns);
                starting_positions.remove(start_indx);
            }
        }

        let mut new_game = Self {
            grid,
            snakes,
            board_size,
            winner: None,
        };
        new_game.put_random_apples(num_players);
        Ok(new_game)
    }

    fn put_random_apples(&mut self, n: usize) {
        let mut avaliable = Vec::new();
        for i in 0..self.grid.len() {
            for j in 0..self.grid[i].len() {
                if matches!(self.grid[i][j], SnakeBlock::Empty) {
                    avaliable.push((i, j));
                }
            }
        }
        let mut n = n;
        if avaliable.is_empty() {
            warn!("No where left to place apples");
            return;
        } else if avaliable.len() < n {
            debug!(
                "{} apples requested but only {} empty blocks",
                n,
                avaliable.len()
            );
            n = avaliable.len();
        }
        let choices = avaliable.choose_multiple(&mut rand::thread_rng(), n);
        for index in choices {
            self.grid[index.0][index.1] = SnakeBlock::Apple;
        }
    }

    pub fn check_for_winner(&mut self) {
        if self.snakes.len() == 1 {
            let first = &self.snakes[0];
            if first.is_dead() {
                info!("Game over");
                self.winner = Some(first.player_id);
            }
            return;
        }
        let mut pot_winner = None;
        for snake in &self.snakes {
            if snake.is_alive() {
                if pot_winner.is_some() {
                    // more than one snake alive -> no winner
                    pot_winner = None;
                    break;
                }
                pot_winner = Some(snake);
            }
        }
        if let Some(winner) = pot_winner {
            info!("Game over. Player {} won", winner.player_id + 1);
            self.winner = Some(winner.player_id);
        }
    }

    /// Attempts to move the character.
    /// Returns true if the snake survived and the game is valid. Else, returns false.
    ///
    /// # Panics
    ///
    /// Panics if the player's snake is empty or type conversions fail.
    pub fn move_character(
        &mut self,
        snake_indx: usize,
        action: SnakeAction,
        new_time: Option<Instant>,
        allow_same_direction: bool,
    ) -> bool {
        if self.winner.is_some() {
            debug!("Trid to move snake {} after winner found", snake_indx);
            return false;
        } else if snake_indx >= self.snakes.len() {
            warn!(
                "Tried to move snake {} but max snake id is {}",
                snake_indx,
                self.snakes.len()
            );
            return false;
        }
        let mut replace_apple = false;

        // probably a skill issue but I need to borrow snake as mutable and then call replace apple
        // but replace aple also needs to borrow self as mutable and only one mutable reference allowed
        // hence I added this scope so snake would be dropped
        {
            let snake = &mut self.snakes[snake_indx];
            if !snake.is_alive() {
                debug!("Tried to move dead snake {snake_indx}");
                return false;
            }
            let delta = action.value();
            if (!allow_same_direction && delta == snake.last_action.value())
                || action.get_opposite().value() == snake.last_action.value()
            {
                return true;
            }

            let front = snake.get_head().unwrap(); // already checked its alive
            let new_x: i64 = i64::try_from(front.0).unwrap() + i64::from(delta.0);
            let new_y: i64 = i64::try_from(front.1).unwrap() + i64::from(delta.1);
            let size = self.board_size.try_into().unwrap();
            if new_x >= size || new_y >= size || new_x < 0 || new_y < 0 {
                debug!("Snake {snake_indx} went out of bounds. Killing snake");
                for st in &snake.squares_taken {
                    self.grid[st.0][st.1] = SnakeBlock::Empty;
                }
                snake.squares_taken = VecDeque::new();
                assert!(snake.is_dead());
                self.check_for_winner();
                return false;
            }
            let new_x = usize::try_from(new_x).unwrap();
            let new_y = usize::try_from(new_y).unwrap();
            match self.grid.get(new_x).unwrap().get(new_y).unwrap() {
                SnakeBlock::Apple => {
                    self.grid[new_x][new_y] = SnakeBlock::SnakeHead(snake.player_id);
                    self.grid[front.0][front.1] = SnakeBlock::SnakeBody(snake.player_id);
                    snake.squares_taken.push_front((new_x, new_y));
                    replace_apple = true;
                }
                SnakeBlock::Empty => {
                    self.grid[new_x][new_y] = SnakeBlock::SnakeHead(snake.player_id);
                    self.grid[front.0][front.1] = SnakeBlock::SnakeBody(snake.player_id);
                    snake.squares_taken.push_front((new_x, new_y));
                    if let Some(old_tail) = snake.squares_taken.pop_back() {
                        self.grid[old_tail.0][old_tail.1] = SnakeBlock::Empty;
                    } else {
                        debug!("Removed from back but got None");
                    }
                }
                SnakeBlock::SnakeBody(p) | SnakeBlock::SnakeHead(p) => {
                    debug!("Snake {snake_indx} ran into snake {p}. Killing snake {snake_indx}");
                    for st in &snake.squares_taken {
                        self.grid[st.0][st.1] = SnakeBlock::Empty;
                    }
                    snake.squares_taken = VecDeque::new();
                    assert!(snake.is_dead());
                    return false;
                }
            }
            if new_time.is_some() {
                snake.time_of_last_action = new_time.unwrap();
            }
            snake.last_action = action;
        }
        if replace_apple {
            self.put_random_apples(1);
        }
        self.check_for_winner();
        true
    }

    #[must_use]
    pub fn get_winner(&self) -> Option<usize> {
        self.winner
    }

    #[must_use]
    pub fn get_grid(&self) -> &Vec<Vec<SnakeBlock>> {
        &self.grid
    }

    #[must_use]
    pub fn get_size(&self) -> usize {
        self.board_size
    }

    #[must_use]
    pub fn get_player(&self, player: usize) -> Option<&SnakePlayer> {
        self.snakes.get(player)
    }

    #[must_use]
    pub fn get_mut_player(&mut self, player: usize) -> Option<&mut SnakePlayer> {
        self.snakes.get_mut(player)
    }

    #[must_use]
    pub fn get_all_players(&self) -> &Vec<SnakePlayer> {
        &self.snakes
    }
}

impl Default for SnakeGame {
    fn default() -> Self {
        Self::new(3, 1, MAX_BOARD_SIZE).unwrap()
    }
}
