use std::cmp::max;

use crate::{
    models::snake::snake_game::{
        BOUNDARY, MAX_BOARD_SIZE, MAX_NUM_OF_REAL_PLAYERS, MAX_NUM_OF_TOTAL_PLAYERS,
        MIN_BOARD_SIZE, MIN_INCR,
    },
    view_model::ViewModel,
};

#[derive(Debug)]
pub struct SnakeSelectionViewModel {}

impl SnakeSelectionViewModel {
    #[must_use]
    pub fn validate_number_of_real_players(&self, number_of_real_players: usize) -> bool {
        number_of_real_players <= MAX_NUM_OF_REAL_PLAYERS
    }

    #[must_use]
    pub fn validate_number_of_bots(&self, number_of_bots: usize) -> bool {
        number_of_bots <= MAX_NUM_OF_TOTAL_PLAYERS
    }

    #[must_use]
    pub fn validate_number_of_total_players(&self, number_of_total_players: usize) -> bool {
        (1..=MAX_NUM_OF_TOTAL_PLAYERS).contains(&number_of_total_players)
    }

    #[must_use]
    pub fn validate_grid_size(&self, grid_size: usize) -> bool {
        (MIN_BOARD_SIZE..=MAX_BOARD_SIZE).contains(&grid_size)
    }

    #[must_use]
    pub fn validate_grid_to_player(
        &self,
        grid_size: usize,
        number_of_total_players: usize,
    ) -> bool {
        grid_size >= self.get_min_grid_size(number_of_total_players)
    }

    #[must_use]
    pub fn validate_all(
        &self,
        grid_size: usize,
        number_of_bots: usize,
        number_of_real_players: usize,
    ) -> bool {
        let number_of_total_players = number_of_real_players + number_of_bots;
        self.validate_number_of_real_players(number_of_real_players)
            && self.validate_number_of_bots(number_of_bots)
            && self.validate_number_of_total_players(number_of_total_players)
            && self.validate_grid_size(grid_size)
            && self.validate_grid_to_player(grid_size, number_of_total_players)
    }

    #[must_use]
    pub fn get_max_total_players(&self) -> usize {
        MAX_NUM_OF_TOTAL_PLAYERS
    }

    #[must_use]
    pub fn get_max_real_players(&self) -> usize {
        MAX_NUM_OF_REAL_PLAYERS
    }

    #[must_use]
    pub fn get_min_grid_size(&self, total_players: usize) -> usize {
        max(total_players * MIN_INCR + 2 * BOUNDARY + 1, MIN_BOARD_SIZE)
    }

    #[must_use]
    pub fn get_max_grid_size(&self) -> usize {
        MAX_BOARD_SIZE
    }
}

impl ViewModel for SnakeSelectionViewModel {
    /// The `SnakeSelectionViewModel` does not hold state.
    fn update(&mut self, _message: crate::app::Message) -> Option<crate::app::Message> {
        None
    }
}
