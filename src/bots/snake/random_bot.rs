use crate::views::snake::snake_home::{self, SnakeBlock};
use rand::Rng;

use super::snake_bot::SnakeBot;

#[derive(Debug)]
pub struct RandomBot {
    move_time: u64,
}

impl SnakeBot for RandomBot {
    fn make_move(&self, _grid: Vec<Vec<SnakeBlock>>) -> (i8, i8) {
        let sign = rand::thread_rng().gen_bool(0.5);
        let index = rand::thread_rng().gen_range(0..=1);
        let mult = if sign { -1 } else { 1 };
        if index == 0 {
            (mult, 0)
        } else {
            (0, mult)
        }
    }

    fn new() -> Self {
        Self {
            move_time: snake_home::MILLIS_BETWEEN_FRAMES * 2,
        }
    }

    fn get_move_time(&self) -> u64 {
        self.move_time
    }
}
