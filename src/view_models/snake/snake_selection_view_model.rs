use std::cmp::max;

use crate::{
    models::snake::snake_game::{
        BOUNDARY, MAX_BOARD_SIZE, MAX_NUM_OF_REAL_PLAYERS, MAX_NUM_OF_TOTAL_PLAYERS, MIN_BOARD_SIZE, MIN_INCR
    },
    view_model::ViewModel,
};

#[derive(Debug)]
pub struct SnakeSelectionViewModel {}

impl SnakeSelectionViewModel {
    pub fn validate_number_of_real_players(&self, nor: usize) -> bool {
        nor <= MAX_NUM_OF_REAL_PLAYERS
    }

    pub fn validate_number_of_bots(&self, nob: usize) -> bool {
        nob <= MAX_NUM_OF_TOTAL_PLAYERS
    }

    pub fn validate_number_of_total_players(&self, notp: usize) -> bool {
        notp >= 1 && notp <= MAX_NUM_OF_TOTAL_PLAYERS
    }

    pub fn validate_grid_size(&self, gs: usize) -> bool {
        gs >= MIN_BOARD_SIZE && gs <= MAX_BOARD_SIZE
    }

    pub fn validate_grid_to_player(&self, gs: usize, notp: usize) -> bool {
        gs / notp >= MIN_INCR
    }

    pub fn validate_all(&self, gs: usize, nob: usize, nor: usize) -> bool {
        let notp = nor + nob;
        return self.validate_number_of_real_players(nor)
            && self.validate_number_of_bots(nob)
            && self.validate_number_of_total_players(notp)
            && self.validate_grid_size(gs)
            && self.validate_grid_to_player(gs, notp);
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
    /// The SnakeSelectionViewModel does not hold state.
    fn update(&mut self, _message: crate::app::Message) -> Option<crate::app::Message> {
        None
    }
}
