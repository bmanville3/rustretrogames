//! Module for containing the Snake Game Player.
use std::{collections::VecDeque, time::Instant};

use super::snake_game::SnakeAction;

pub const MAX_QUEUE_SIZE: usize = 3;

/// Snake player in the [`crate::models::snake::snake_game::SnakeGame`].
#[derive(Clone, Debug)]
pub struct SnakePlayer {
    pub is_bot: bool,
    pub squares_taken: VecDeque<(usize, usize)>,
    pub last_action: SnakeAction,
    pub time_of_last_action: Instant,
    pub player_id: usize,
    move_queue: VecDeque<SnakeAction>,
}

impl SnakePlayer {
    /// Creates a new player at the specified coordinates.
    #[must_use]
    pub fn new(
        is_bot: bool,
        x: usize,
        y: usize,
        player_id: usize,
        starting_action: SnakeAction,
    ) -> Self {
        let mut squares_taken = VecDeque::new();
        squares_taken.push_front((x, y));
        Self {
            is_bot,
            squares_taken,
            last_action: starting_action,
            time_of_last_action: Instant::now(),
            player_id,
            move_queue: VecDeque::with_capacity(MAX_QUEUE_SIZE),
        }
    }

    /// Check if the snake is still alive.
    #[must_use]
    pub fn is_alive(&self) -> bool {
        !self.squares_taken.is_empty()
    }

    #[must_use]
    pub fn is_dead(&self) -> bool {
        !self.is_alive()
    }

    /// Returns the head of the snake if it is still alive.
    #[must_use]
    pub fn get_head(&self) -> Option<(usize, usize)> {
        self.squares_taken.front().copied()
    }

    #[must_use]
    pub fn get_name(&self) -> String {
        if self.is_bot {
            return format!("Bot {}", self.player_id + 1);
        }
        format!("Player {}", self.player_id + 1)
    }

    pub fn pop_next_move(&mut self) -> SnakeAction {
        match self.move_queue.pop_front() {
            Some(nm) => nm,
            None => self.last_action.clone(),
        }
    }

    pub fn push_move(&mut self, action: SnakeAction) -> bool {
        #[cfg(debug_assertions)]
        {
            assert!(
                (self.move_queue.len() <= MAX_QUEUE_SIZE),
                "Crash and burn. Queue was too big"
            );
        }
        let at_max_moves = self.move_queue.len() == MAX_QUEUE_SIZE;
        let act_val = action.value();
        let back_would_be_invalid = if let Some(back) = self.move_queue.back() {
            back.value() == act_val || back.get_opposite().value() == act_val
        } else {
            false
        };
        let front_would_be_invalid = if self.move_queue.is_empty() {
            self.last_action.value() == act_val
                || self.last_action.get_opposite().value() == act_val
        } else {
            false
        };
        if at_max_moves || back_would_be_invalid || front_would_be_invalid {
            return false;
        }
        self.move_queue.push_back(action);
        true
    }
}
