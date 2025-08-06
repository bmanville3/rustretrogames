use crate::models::snake::{
    snake_bot::SnakeBot,
    snake_game::{self, PartialSnakeGame, SnakeAction},
};

#[derive(Debug)]
pub struct RandomBot {
    player_indx: usize,
}

impl SnakeBot for RandomBot {
    fn make_move(&self, _game_state: PartialSnakeGame) -> SnakeAction {
        SnakeAction::get_random_action()
    }

    fn new(player_indx: usize) -> Self {
        Self { player_indx }
    }

    fn get_move_time() -> u64 {
        snake_game::MILLIS_BETWEEN_FRAMES * 2
    }

    fn get_player_index(&self) -> usize {
        self.player_indx
    }
}
