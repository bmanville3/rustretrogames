use std::{fs::{self, File}, io::{BufReader, BufWriter}, path::PathBuf};

use chrono::Local;
use log::{debug, error, info};
use once_cell::sync::Lazy;
use rand::{Rng, seq::SliceRandom, thread_rng};

use crate::{deep::{layer::StatelessLayer, linear::{StatelessLinear, WeightInit}, relu::StatelessReLU, sequential::{SequentialLayer, StatelessSequential}}, models::snake::{
    snake_bot::{SnakeBot, SnakeBotType},
    snake_game::{MAX_BOARD_SIZE, MAX_NUM_OF_TOTAL_PLAYERS, SnakeAction, SnakeBlock, SnakeError, SnakeGame},
}, rl::{double_dql::DoubleDQLTrainer, environment::Environment}};

const SNAKE_CHANNEL_SIZE: usize = MAX_BOARD_SIZE * MAX_BOARD_SIZE + SnakeAction::VARIANTS.len() + 2;
const OTHER_CHANNEL_SIZE: usize = MAX_BOARD_SIZE * MAX_BOARD_SIZE;
const FULL_STATE_SIZE: usize = SNAKE_CHANNEL_SIZE * MAX_NUM_OF_TOTAL_PLAYERS + 2 * OTHER_CHANNEL_SIZE;


static NEWEST_MODEL: Lazy<StatelessSequential> = Lazy::new(get_newest_snake_model);

fn get_newest_snake_model() -> StatelessSequential {
    let models_dir = PathBuf::from("trained_models");
    std::fs::create_dir_all(&models_dir).unwrap();

    // Look for an existing model JSON
    let newest_model_file = fs::read_dir(&models_dir)
        .unwrap()
        .filter_map(|entry| entry.ok())
        .filter(|entry| {
            let path = entry.path();
            path.is_file() && path.extension().map(|ext| ext == "json").unwrap_or(false)
        })
        .max_by_key(|entry| entry.path().file_stem().map(|s| s.to_os_string()).unwrap_or_default()); 

    let model_path = if let Some(file) = newest_model_file {
        file.path()
    } else {
        // No model found, create a new timestamped file
        let timestamp = Local::now().format("%Y-%m-%d_%H-%M-%S").to_string();
        models_dir.join(format!("model_{}.json", timestamp))
    };

    if model_path.exists() {
        // Load existing model
        info!("Loading existing Snake model from {:?}", model_path);
        let file = File::open(&model_path).unwrap();
        let reader = BufReader::new(file);
        serde_json::from_reader(reader).unwrap()
    } else {
        // Create new model
        info!("No existing Snake model found. Creating a new one at {:?}", model_path);
        let model = SnakeGameEnv::get_model();

        // Save it immediately
        let file = File::create(&model_path).unwrap();
        let writer = BufWriter::new(file);
        serde_json::to_writer_pretty(writer, &model).unwrap();

        model
    }
}

#[derive(Clone, Debug)]
pub struct SnakeGameVec {
    pub vec: Vec<f32>,
}

impl SnakeGameVec {
    fn create_new_player_channel(game: &SnakeGame, player_id: usize) -> Vec<f32> {
        let all_players = game.get_all_players();
        let mut channel: Vec<f32> = vec![0.0; SNAKE_CHANNEL_SIZE];
        if player_id < all_players.len() {
            let player = &all_players[player_id];
            if player.is_alive() {
                for (row, col) in &player.squares_taken {
                    channel[row * MAX_BOARD_SIZE + col] = 1.0
                }
                let movement = match player.last_action {
                    SnakeAction::Up => (1.0, 0.0, 0.0, 0.0),
                    SnakeAction::Down => (0.0, 1.0, 0.0, 0.0),
                    SnakeAction::Left => (0.0, 0.0, 1.0, 0.0),
                    SnakeAction::Right => (0.0, 0.0, 0.0, 1.0),
                };
                channel[SNAKE_CHANNEL_SIZE - 6] = movement.0;
                channel[SNAKE_CHANNEL_SIZE - 5] = movement.1;
                channel[SNAKE_CHANNEL_SIZE - 4] = movement.2;
                channel[SNAKE_CHANNEL_SIZE - 3] = movement.3;
                let (head_r, head_c) = match player.get_head() {
                    Some(h) => h,
                    None => {
                        error!("Play was not dead but did not have a head");
                        return channel;
                    }
                };
                // scale to stabalize learning (particularly prevent exploding gradient)
                channel[SNAKE_CHANNEL_SIZE - 2] = head_r as f32 / MAX_BOARD_SIZE as f32;
                channel[SNAKE_CHANNEL_SIZE - 1] = head_c as f32 / MAX_BOARD_SIZE as f32;
            }
        }
        channel
    }

    pub fn from_snake_game(game: &SnakeGame, vip: usize) -> Self {
        let grid = game.get_grid();
        let actual_size = grid.len();
        let mut full_state = Vec::with_capacity(SNAKE_CHANNEL_SIZE);

        // build the player channels
        // ALWAYS but the vip first - this is what the bot is play
        full_state.extend(SnakeGameVec::create_new_player_channel(game, vip));
        for i in 0..MAX_NUM_OF_TOTAL_PLAYERS {
            if i == vip {
                continue;
            }
            full_state.extend(SnakeGameVec::create_new_player_channel(game, i));
        }

        // build the valid area channel
        let mut invalid_area_channel = vec![0.0; OTHER_CHANNEL_SIZE];
        for row in 0..MAX_BOARD_SIZE {
            for col in 0..MAX_BOARD_SIZE {
                if row >= actual_size || col >= actual_size {
                    invalid_area_channel[row * MAX_BOARD_SIZE + col] = 1.0;
                }
            }
        }

        full_state.extend(invalid_area_channel);

        // build the apple channel
        let mut apple_channel = vec![0.0; OTHER_CHANNEL_SIZE];
        for row in 0..actual_size {
            for col in 0..actual_size {
                match grid[row][col] {
                    SnakeBlock::Apple => apple_channel[row * MAX_BOARD_SIZE + col] = 1.0,
                    _ => (),
                }
            }
        }
        full_state.extend(apple_channel);
        debug_assert_eq!(full_state.len(), FULL_STATE_SIZE);
        SnakeGameVec { vec: full_state }
    }
}

impl Into<Vec<f32>> for SnakeGameVec {
    fn into(self) -> Vec<f32> {
        self.vec
    }
}

pub struct SnakeGameEnv {
    game: SnakeGame,
    bots: Vec<Box<dyn SnakeBot>>,
    number_of_enemies: usize,
}

impl SnakeGameEnv {
    pub fn new() -> Result<Self, SnakeError> {
        let game = SnakeGame::new(1, MAX_BOARD_SIZE)?;
        let bots = Vec::new();
        let number_of_enemies = 0;
        Ok(Self { game, bots, number_of_enemies })
    }

    pub fn get_model() -> StatelessSequential {
        let mut seq = StatelessSequential::new();
        seq.add(SequentialLayer::Linear(StatelessLinear::new(FULL_STATE_SIZE, FULL_STATE_SIZE / 64, WeightInit::He)));
        seq.add(SequentialLayer::ReLU(StatelessReLU::new()));
        seq.add(SequentialLayer::Linear(StatelessLinear::new(FULL_STATE_SIZE / 64, FULL_STATE_SIZE / 128, WeightInit::He)));
        seq.add(SequentialLayer::ReLU(StatelessReLU::new()));
        seq.add(SequentialLayer::Linear(StatelessLinear::new(FULL_STATE_SIZE / 128, FULL_STATE_SIZE / 256, WeightInit::He)));
        seq.add(SequentialLayer::ReLU(StatelessReLU::new()));
        seq.add(SequentialLayer::Linear(StatelessLinear::new(FULL_STATE_SIZE / 256, SnakeAction::VARIANTS.len(), WeightInit::He)));
        seq
    }
}

impl Environment for SnakeGameEnv {
    type State = SnakeGameVec;
    type Action = SnakeAction;

    fn reset(&mut self) -> Self::State {
        debug!("Reseting state");
        let mut rng = thread_rng();
        self.number_of_enemies = (self.number_of_enemies + 1) % (MAX_NUM_OF_TOTAL_PLAYERS - 1);
        let min_board_size = SnakeGame::get_min_grid_size(self.number_of_enemies + 1);
        self.game = match SnakeGame::new(self.number_of_enemies + 1, rng.gen_range(min_board_size..=MAX_BOARD_SIZE)) {
            Ok(sg) => sg, 
            Err(e) => {
                error!("Error creating new snake game. Cannot reset game: {:#?}", e);
                self.game.clone()
            }
        };
        self.bots = Vec::with_capacity(self.number_of_enemies);
        let bot_type = [SnakeBotType::RandomMoveBot, SnakeBotType::ClosestAppleBot, SnakeBotType::KillerBot].choose(&mut rng).unwrap().clone();
        for i in 0..self.number_of_enemies {
            self.bots.push(bot_type.make_new_bot(i + 1));
        }
        SnakeGameVec::from_snake_game(&self.game, 0)
    }

    fn step(&mut self, action: &Self::Action) -> (Self::State, f32, bool) {
        if self.game.is_over() {
            error!("Tried to step over a dead game");
            return (SnakeGameVec::from_snake_game(&self.game, 0), 0.0, true)
        }

        let size_before = {
            let us_before = self.game.get_player(0).unwrap();
            if us_before.is_dead() {
                error!("Tried to continue to step after player was already dead");
                return (SnakeGameVec::from_snake_game(&self.game, 0), 0.0, true)
            }
            us_before.squares_taken.len()
        };

        let enemies_alive_before: Vec<usize> = self.game.get_all_players().iter().filter(|s| s.is_alive() && s.player_id != 0).map(|s| s.player_id).collect();
        let num_enemies_alive_before = enemies_alive_before.len();

        for i in enemies_alive_before {
            // the bot index in self.bots equals player_indx - 1
            self.game.add_move_to_snake(i, self.bots[i - 1].make_move(&self.game));
        }
        // first index is always our main guy
        self.game.add_move_to_snake(0, action.clone());
        self.game.move_all_characters();

        if self.game.get_player(0).unwrap().is_dead() {
            return (SnakeGameVec::from_snake_game(&self.game, 0), -10.0, true) // get a -10 for dying -> very bad
        }

        let mut reward = 0.01; // reward for surviving

        let num_enemies_alive_after = self.game.get_all_players().iter().filter(|s| s.is_alive() && s.player_id != 0).count();

        reward += (num_enemies_alive_before - num_enemies_alive_after) as f32 * 2.0; // get +2 for every dead enemy

        let size_after = self.game.get_player(0).unwrap().squares_taken.len();
        reward += (size_after - size_before) as f32 * 1.0; // get a +1 for eating an apple

        if let Some(winner) = self.game.get_winner() {
            if winner == 0 {
                // big reward for winning
                reward += 10.0;
            }
        }

        (SnakeGameVec::from_snake_game(&self.game, 0), reward, self.game.is_over())
    }

    fn get_action_mask(&self) -> Vec<bool> {
        // all actions are always valid
        vec![true; SnakeAction::VARIANTS.len()]
    }

    fn all_actions() -> Vec<Self::Action> {
        SnakeAction::VARIANTS.to_vec()
    }

    fn action_to_index(action: &Self::Action) -> usize {
        SnakeAction::VARIANTS
            .iter()
            .position(|a| a == action)
            .unwrap_or_else(|| {
                error!("Action not in VARIANTS: {:?}", action);
                0
            })
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

    pub fn train_new_bot() {
        let mut env = match SnakeGameEnv::new() {
            Ok(s) => s,
            Err(e) => {
                panic!("Could not make new snake game env: {:#?}", e);
            }
        };

        let gamma = 0.95;
        let epsilon = 0.99;
        let epsilon_decay = 0.999;
        let epsilon_min = 0.05;
        let learning_rate = 0.0001;
        let buffer_capacity = 10_000;

        let dql = NEWEST_MODEL.clone();

        let mut trainer = DoubleDQLTrainer::<SnakeGameEnv, StatelessSequential>::new(
            dql,
            gamma,
            epsilon,
            epsilon_decay,
            epsilon_min,
            learning_rate,
            buffer_capacity,
        );

        let episodes = 50000;
        let batch_size = 64;
        let max_moves = 500;
        let update_target_after_steps = 1000;

        let models_dir = PathBuf::from("trained_models");
        std::fs::create_dir_all(&models_dir).unwrap();

        // Generate a timestamped filename
        let timestamp = Local::now().format("%Y-%m-%d_%H-%M-%S").to_string();
        let file_path = models_dir.join(format!("model_{}.json", timestamp));

        trainer.train(&mut env, episodes, batch_size, max_moves, update_target_after_steps, Some(file_path.as_path()));
    }
}

impl SnakeBot for DQLBot {
    fn make_move(&self, game_state: &SnakeGame) -> SnakeAction {
        let (index, value) = NEWEST_MODEL.forward(&SnakeGameVec::from_snake_game(game_state, self.player_indx).vec).0.iter()
                            .enumerate()
                            .fold((0, f32::NEG_INFINITY), |(current_max_idx, current_max_val), (idx, &val)| {
                                if val > current_max_val {
                                    (idx, val)
                                } else {
                                    (current_max_idx, current_max_val)
                                }
                            });
        if value.is_nan() || value.is_infinite() {
            error!("Q value from bot was invalid");
            return SnakeAction::get_random_action();
        }
        let action = &SnakeGameEnv::all_actions()[index];
        action.clone()
    }

    fn get_player_index(&self) -> usize {
        self.player_indx
    }

    fn warmup(&mut self) {
        let _ = Lazy::force(&NEWEST_MODEL);
        info!("DQL Bot is warmed up")
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn snake_training_work() {
        let _ = env_logger::builder().is_test(true).try_init();
        DQLBot::train_new_bot();
    }
}

