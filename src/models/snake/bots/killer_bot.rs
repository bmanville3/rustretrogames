use log::error;
use rand::Rng;

use crate::models::snake::{
    snake_bot::SnakeBot,
    snake_game::{PartialSnakeGame, SnakeAction},
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
        game_state: &PartialSnakeGame,
        other_index: usize,
    ) -> Option<(usize, usize)> {
        let rows = game_state.grid.len();
        let cols = game_state.grid[0].len();

        let other_head = match game_state.snake_heads[other_index] {
            Some(pos) => pos,
            None => return None,
        };
        let other_last_move = *game_state
            .last_movements
            .get(other_index)
            .unwrap_or(&(0, 0));
        let nx_is = other_head.0 as isize + other_last_move.0 as isize;
        let ny_is = other_head.1 as isize + other_last_move.1 as isize;
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
        game_state: &PartialSnakeGame,
        my_head: (usize, usize),
    ) -> Option<((usize, usize), usize)> {
        let mut best: Option<((usize, usize), usize, usize)> = None; // (cell, idx, distance)

        for i in 0..game_state.snake_heads.len() {
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
        game_state: &PartialSnakeGame,
        head: (usize, usize),
        max_dist: usize,
    ) -> Option<(i8, i8)> {
        self.bfs_towards_goal(
            game_state,
            head,
            &|x, y| {
                matches!(
                    game_state.grid[x][y],
                    crate::models::snake::snake_game::SnakeBlock::Apple
                )
            },
            Some(max_dist),
        )
    }
}

impl SnakeBot for KillerBot {
    fn make_move(&self, game_state: PartialSnakeGame) -> SnakeAction {
        let our_head = match game_state.snake_heads[self.player_indx] {
            Some(pos) => pos,
            None => {
                error!("Snake head not found for player {}", self.player_indx);
                return SnakeAction::get_random_action();
            }
        };

        let last_move = *game_state
            .last_movements
            .get(self.player_indx)
            .unwrap_or(&(0, 0));

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

        // add a tiny bit of randomness so bots dont just go side-by-side up and down the board
        if rand::thread_rng().gen::<f32>() < 0.95 {
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

        for alt in SnakeAction::VARIANTS {
            let mv = alt.value();
            if self.is_move_valid_and_safe(mv, our_head, last_move, &game_state) {
                return alt.clone();
            }
        }

        SnakeAction::get_random_action()
    }

    fn get_player_index(&self) -> usize {
        self.player_indx
    }
}
