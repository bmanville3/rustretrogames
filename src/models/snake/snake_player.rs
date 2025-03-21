//! Module for containing the Snake Game Player.
use std::{collections::VecDeque, time::Instant};

use super::snake_game::SnakeAction;

/// Snake player in the [crate::models::snake::snake_game::SnakeGame].
#[derive(Clone, Debug)]
pub struct SnakePlayer {
    pub is_bot: bool,
    pub squares_taken: VecDeque<(usize, usize)>,
    pub last_action: SnakeAction,
    pub time_of_last_action: Instant,
    pub player_id: usize,
}

impl SnakePlayer {
    /// Creates a new player at the specified coordinates.
    pub fn new(is_bot: bool, x: usize, y: usize, player_id: usize, starting_action: SnakeAction) -> Self {
        let mut squares_taken = VecDeque::new();
        squares_taken.push_front((x, y));
        Self {
            is_bot,
            squares_taken,
            last_action: starting_action,
            time_of_last_action: Instant::now(),
            player_id,
        }
    }

    /// Check if the snake is still alive.
    pub fn is_alive(&self) -> bool {
        self.squares_taken.len() != 0
    }

    pub fn is_dead(&self) -> bool {
        !self.is_alive()
    }

    /// Returns the head of the snake if it is still alive.
    pub fn get_head(&self) -> Option<(usize, usize)> {
        self.squares_taken.get(0).copied()
    }

    pub fn get_name(&self) -> String {
        if self.is_bot {
            return format!("Bot {}", self.player_id + 1);
        }
        return format!("Player {}", self.player_id + 1);
    }
}
