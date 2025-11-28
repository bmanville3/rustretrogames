use log::error;
use rand::{Rng, seq::SliceRandom};

use crate::models::snake::{
    snake_bot::SnakeBot,
    snake_game::{SnakeGame, SnakeAction},
};

#[derive(Debug)]
pub struct KillerBot {
    player_indx: usize,
}

impl KillerBot {
    pub fn new(player_indx: usize) -> Self {
        Self { player_indx }
    }

    fn front_cell_of_snake(
        &self,
        game_state: &SnakeGame,
        other_index: usize,
    ) -> Option<(usize, usize)> {
        let rows = game_state.get_size();
        let cols = game_state.get_size();

        let other_snake = match game_state.get_all_players().get(other_index) {
            Some(pos) => pos,
            None => return None,
        };
        let other_head = match other_snake.get_head() {
            Some(h) => h,
            None => return None,
        };
        let other_last_move = other_snake.last_action.value();
        // predict 2 moves ahead and not just one - trying not to kill ourselves
        let nx_is = other_head.0 as isize +  2 * other_last_move.0 as isize;
        let ny_is = other_head.1 as isize + 2 * other_last_move.1 as isize;
        if nx_is < 0 || ny_is < 0 {
            return None;
        }
        let nx = nx_is as usize;
        let ny = ny_is as usize;
        if nx >= rows || ny >= cols {
            return None;
        }
        Some((nx, ny))
    }

    fn find_nearest_opponent_front(
        &self,
        game_state: &SnakeGame,
        my_head: (usize, usize),
    ) -> Option<((usize, usize), usize)> {
        let mut best: Option<((usize, usize), usize, usize)> = None; // (cell, idx, distance)

        for i in 0..game_state.get_number_of_players() {
            if i == self.player_indx {
                continue;
            }

            let try_cell = self.front_cell_of_snake(game_state, i);
            if let Some(cell) = try_cell {
                let manhattan = (cell.0 as isize - my_head.0 as isize).abs()
                    + (cell.1 as isize - my_head.1 as isize).abs();
                let manhattan = manhattan as usize;

                let consider = match best {
                    Some((_, _, best_d)) => manhattan < best_d,
                    None => true,
                };
                if consider {
                    best = Some((cell, i, manhattan));
                }
            }
        }

        best.map(|(cell, i, _)| (cell, i))
    }

    fn find_nearby_apple_step(
        &self,
        game_state: &SnakeGame,
        head: (usize, usize),
        max_dist: usize,
    ) -> Option<(i8, i8)> {
        self.bfs_towards_goal(
            game_state,
            head,
            &|x, y| {
                matches!(
                    game_state.get_grid()[x][y],
                    crate::models::snake::snake_game::SnakeBlock::Apple
                )
            },
            Some(max_dist),
        )
    }
}

impl SnakeBot for KillerBot {
    fn make_move(&self, game_state: &SnakeGame) -> SnakeAction {
        let our_snake = match game_state.get_all_players().get(self.player_indx) {
            Some(pos) => pos,
            None => {
                error!("Snake head not found for player {}", self.player_indx);
                return SnakeAction::get_random_action();
            }
        };

        let last_move = our_snake.last_action.value();
        let our_head = match our_snake.get_head() {
            Some(h) => h,
            None => {
                error!("Tried to move a dead snake");
                return SnakeAction::get_random_action()
            }
        };
        let mut rng = rand::thread_rng();
        // add a tiny bit of randomness so bots dont just go side-by-side up and down the board
        if rng.gen::<f32>() < 0.90 {
            // prefer eating an apple if its within 3 steps
            if let Some(step_to_apple) = self.find_nearby_apple_step(&game_state, our_head, 3) {
                let action =
                    SnakeAction::get_enum_variant_from_values(step_to_apple.0, step_to_apple.1);
                if let Ok(a) = action {
                    return a;
                } else {
                    error!(
                        "Apple BFS returned invalid move: {:#?}. Trying to find head instead",
                        action.err()
                    );
                }
            }

            if let Some((target_cell, _opponent_idx)) =
                self.find_nearest_opponent_front(&game_state, our_head)
            {
                let maybe_step = self.bfs_towards_goal(
                    &game_state,
                    our_head,
                    &|x, y| (x, y) == target_cell,
                    None,
                );

                if let Some(step) = maybe_step {
                    return match SnakeAction::get_enum_variant_from_values(step.0, step.1) {
                        Ok(action) => action,
                        Err(e) => {
                            error!("intercept BFS returned bad move: {:#?}", e);
                            SnakeAction::get_random_action()
                        }
                    };
                }
            }
        }

        let mut variants: Vec<_> = SnakeAction::VARIANTS.to_vec();
        variants.shuffle(&mut rng);

        for alt in variants {
            let mv = alt.value();
            if self.is_move_valid_and_safe(mv, our_head, last_move, game_state) {
                return alt;
            }
        }

        SnakeAction::get_random_action()
    }

    fn get_player_index(&self) -> usize {
        self.player_indx
    }
}
