use crate::{models::snake::{
    snake_bot::SnakeBot,
    snake_game::{MAX_BOARD_SIZE, PartialSnakeGame, SnakeAction, SnakeGame},
}, rl::environment::Environment};

#[derive(Clone, Debug)]
pub struct SnakeGameVec {
    pub vec: Vec<f32>,
}

impl SnakeGameVec {
    pub fn from_partial_snake_game(partial_game: PartialSnakeGame) -> Self {
        todo!();
    }
}

impl Into<Vec<f32>> for SnakeGameVec {
    fn into(self) -> Vec<f32> {
        self.vec
    }
}

struct SnakeGameEnv {
    game: SnakeGame,

}

impl Environment for SnakeGameEnv {
    type State = SnakeGameVec;
    type Action = SnakeAction;

    fn reset(&mut self) -> Self::State {
        self.game = SnakeGame::new(1, 0, MAX_BOARD_SIZE);
        SnakeGameVec::from_partial_snake_game(PartialSnakeGame::from_full(self.game.clone()))
    }

    fn step(&mut self, action: &Self::Action) -> (Self::State, f32, bool) {
        let next = self.transition(&self.state, action);
        self.state = next.clone();

        if self.is_goal(&next) {
            return (next, 1.0, true);
        }
        if self.is_death(&next) {
            return (next, -1.0, true);
        }

        (next, -0.01, false)
    }

    fn get_action_mask(&self) -> Vec<bool> {
        TestEnv::get_action_mask_of_state(&self.state)
    }

    fn all_actions() -> Vec<Self::Action> {
        vec![
            TestAction::Left,
            TestAction::Right,
            TestAction::Up,
            TestAction::Down,
        ]
    }

    fn action_to_index(action: &Self::Action) -> usize {
        match action {
            TestAction::Left => 0,
            TestAction::Right => 1,
            TestAction::Up => 2,
            TestAction::Down => 3,
        }
    }
}

#[derive(Debug)]
pub struct DQLBot {
    player_indx: usize,
}

impl DQLBot {
    pub fn new(player_indx: usize) -> Self {
        Self { player_indx }
    }

    fn train_new_bot() {

    }
}

impl SnakeBot for DQLBot {
    fn make_move(&self, _game_state: PartialSnakeGame) -> SnakeAction {
        SnakeAction::get_random_action()
    }

    fn get_player_index(&self) -> usize {
        self.player_indx
    }
}
