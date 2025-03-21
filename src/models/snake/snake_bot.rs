use super::{
    bots::random_snake_bot::RandomBot,
    snake_game::{SnakeAction, SnakeGame},
};

pub trait SnakeBot {
    fn new(player_indx: usize) -> Self;

    fn get_move_time() -> u64;

    fn make_move(&self, game_state: SnakeGame) -> SnakeAction;

    fn get_player_index(&self) -> usize;
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SnakeBotType {
    RandomMoveBot,
}

// this solution doesn't scale well but the number of bot types will be small so it works
impl SnakeBotType {
    pub const VALUES: [Self; 1] = [Self::RandomMoveBot];

    /// Amount of time the bot must wait before moving again.
    #[must_use]
    pub fn get_move_time(&self) -> u64 {
        match self {
            SnakeBotType::RandomMoveBot => RandomBot::get_move_time(),
        }
    }

    #[must_use]
    pub fn make_new_bot(&self, player_indx: usize) -> impl SnakeBot {
        match self {
            SnakeBotType::RandomMoveBot => RandomBot::new(player_indx),
        }
    }
}

impl std::fmt::Display for SnakeBotType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SnakeBotType::RandomMoveBot => write!(f, "Randomly Moving Bot"),
        }
    }
}
