use super::{bots::random_snake_bot::RandomBot, snake_model::SnakeBlock};

pub trait SnakeBot {
    fn new() -> Self;

    fn make_move(&self, grid: Vec<Vec<SnakeBlock>>) -> (i8, i8);

    fn get_move_time(&self) -> u64;
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SnakeBotType {
    RandomMoveBot,
}

// this solution doesn't scale well but the number of bot types will be small so it works
impl SnakeBotType {
    pub const VALUES: [Self; 1] = [Self::RandomMoveBot];
}

impl std::fmt::Display for SnakeBotType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SnakeBotType::RandomMoveBot => write!(f, "Randomly Moving Bot"),
        }
    }
}

#[must_use]
pub fn make_new_bot(bot_type: &SnakeBotType) -> impl SnakeBot {
    match bot_type {
        SnakeBotType::RandomMoveBot => RandomBot::new(),
    }
}
