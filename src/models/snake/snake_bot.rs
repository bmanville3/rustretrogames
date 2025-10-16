use std::collections::VecDeque;

use crate::models::snake::{bots::killer_bot::KillerBot, snake_game::SnakeBlock};

use super::{
    bots::move_to_closest_apple_bot::MoveToClosestAppleBot,
    bots::random_snake_bot::RandomBot,
    snake_game::{PartialSnakeGame, SnakeAction},
};

pub trait SnakeBot: Send + Sync {
    fn make_move(&self, game_state: PartialSnakeGame) -> SnakeAction;

    fn get_player_index(&self) -> usize;

    fn is_move_valid_and_safe(
        &self,
        next_move: (i8, i8),
        head: (usize, usize),
        last_move: (i8, i8),
        game_state: &PartialSnakeGame,
    ) -> bool {
        if (next_move.0, next_move.1) == (-last_move.0, -last_move.1) {
            return false;
        }
        if SnakeAction::get_enum_variant_from_tuple(next_move).is_err() {
            return false;
        }

        let new_x = head.0 as isize + next_move.0 as isize;
        let new_y = head.1 as isize + next_move.1 as isize;

        if new_x < 0 || new_x >= game_state.grid.len() as isize {
            return false;
        }
        if new_y < 0 || new_y >= game_state.grid[0].len() as isize {
            return false;
        }

        match game_state.grid[new_x as usize][new_y as usize] {
            SnakeBlock::Empty | SnakeBlock::Apple => true,
            _ => false,
        }
    }

    /// Will return the next move to take to get towards the goal (if possible).
    /// This move is guaranteed to be valid and safe if present.
    fn bfs_towards_goal(
        &self,
        game_state: &PartialSnakeGame,
        head: (usize, usize),
        is_goal: &(dyn Fn(usize, usize) -> bool),
        max_depth: Option<usize>,
    ) -> Option<(i8, i8)> {
        if is_goal(head.0, head.1) {
            return None;
        }

        let rows = game_state.grid.len();
        let cols = game_state.grid[0].len();

        let mut visited = vec![false; rows * cols];
        let mut queue = VecDeque::new();
        let mut parent = vec![None; rows * cols]; // store parent positions
        let mut depth: Vec<Option<usize>> = vec![None; rows * cols];
        queue.push_back(head);
        visited[head.0 + head.1 * rows] = true;

        let directions: Vec<(i8, i8)> = SnakeAction::VARIANTS.iter().map(|a| a.value()).collect();

        while let Some((x, y)) = queue.pop_front() {
            let idx = x + y * rows;
            let cur_depth = depth[idx].unwrap_or(0);
            if is_goal(x, y) {
                // backtrack to find first move
                let mut cur = (x, y);
                while let Some(p) = parent[cur.0 + cur.1 * rows] {
                    if p == head {
                        let dx = cur.0 as isize - head.0 as isize;
                        let dy = cur.1 as isize - head.1 as isize;
                        return Some((dx as i8, dy as i8));
                    }
                    cur = p;
                }
            }

            if let Some(max_d) = max_depth {
                if cur_depth >= max_d {
                    continue;
                }
            }

            for &(dx, dy) in &directions {
                if !self.is_move_valid_and_safe((dx, dy), (x, y), (0, 0), game_state) {
                    continue;
                }
                let nx = (x as isize + dx as isize) as usize;
                let ny = (y as isize + dy as isize) as usize;
                let nidx = nx + ny * rows;
                if visited[nidx] {
                    continue;
                }
                visited[nidx] = true;
                parent[nidx] = Some((x, y));
                depth[nidx] = Some(cur_depth + 1);
                queue.push_back((nx, ny));
            }
        }

        None
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SnakeBotType {
    ClosestAppleBot,
    KillerBot,
    RandomMoveBot,
}

// this solution doesn't scale well but the number of bot types will be small so it works
impl SnakeBotType {
    pub const VALUES: [Self; 3] = [Self::RandomMoveBot, Self::ClosestAppleBot, Self::KillerBot];

    #[must_use]
    pub fn make_new_bot(&self, player_indx: usize) -> Box<dyn SnakeBot> {
        match self {
            SnakeBotType::ClosestAppleBot => Box::new(MoveToClosestAppleBot::new(player_indx)),
            SnakeBotType::KillerBot => Box::new(KillerBot::new(player_indx)),
            SnakeBotType::RandomMoveBot => Box::new(RandomBot::new(player_indx)),
        }
    }
}

impl std::fmt::Display for SnakeBotType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SnakeBotType::ClosestAppleBot => write!(f, "BFS Closest Apple Finding Bot"),
            SnakeBotType::KillerBot => write!(f, "BFS Killer Bot"),
            SnakeBotType::RandomMoveBot => write!(f, "Randomly Moving Bot"),
        }
    }
}
