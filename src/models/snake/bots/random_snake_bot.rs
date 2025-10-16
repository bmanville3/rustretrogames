use crate::models::snake::{
    snake_bot::SnakeBot,
    snake_game::{PartialSnakeGame, SnakeAction},
};

#[derive(Debug)]
pub struct RandomBot {
    player_indx: usize,
}

impl RandomBot {
    pub fn new(player_indx: usize) -> Self {
        Self { player_indx }
    }
}

impl SnakeBot for RandomBot {
    fn make_move(&self, _game_state: PartialSnakeGame) -> SnakeAction {
        SnakeAction::get_random_action()
    }

    fn get_player_index(&self) -> usize {
        self.player_indx
    }
}
