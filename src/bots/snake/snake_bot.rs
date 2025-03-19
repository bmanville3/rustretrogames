use crate::views::snake::snake_home::SnakeBlock;

pub trait SnakeBot {
    fn new() -> Self;

    fn make_move(&self, grid: Vec<Vec<SnakeBlock>>) -> (i8, i8);

    fn get_move_time(&self) -> u64;
}
