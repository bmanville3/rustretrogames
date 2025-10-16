use log::error;

use crate::models::snake::{
    snake_bot::SnakeBot,
    snake_game::{PartialSnakeGame, SnakeAction, SnakeBlock},
};

#[derive(Debug)]
pub struct MoveToClosestAppleBot {
    player_indx: usize,
}

impl MoveToClosestAppleBot {
    pub fn new(player_indx: usize) -> Self {
        Self { player_indx }
    }

    fn find_closest_reachable_apple(
        &self,
        game_state: &PartialSnakeGame,
        head: (usize, usize),
    ) -> Option<(i8, i8)> {
        self.bfs_towards_goal(
            game_state,
            head,
            &|x, y| matches!(game_state.grid[x][y], SnakeBlock::Apple),
            None,
        )
    }
}

impl SnakeBot for MoveToClosestAppleBot {
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

        // Compute the preferred direction toward the closest apple
        let target_move_opt = self.find_closest_reachable_apple(&game_state, our_head);

        // Check if the move is present and valid
        if let Some(target_move) = target_move_opt {
            return match SnakeAction::get_enum_variant_from_values(target_move.0, target_move.1) {
                Ok(action) => action,
                Err(e) => {
                    error!("BFS returned a bad move: {:#?}", e);
                    SnakeAction::get_random_action()
                }
            };
        }

        for alt_move in SnakeAction::VARIANTS {
            if self.is_move_valid_and_safe(alt_move.value(), our_head, last_move, &game_state) {
                return alt_move.clone();
            }
        }

        // no where to go
        return SnakeAction::get_random_action();
    }

    fn get_player_index(&self) -> usize {
        self.player_indx
    }
}
