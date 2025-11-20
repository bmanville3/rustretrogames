//! Module to perform the Double Deep Q Learning algorithm.
use core::f32;
use std::collections::VecDeque;
use std::fs::{self, File};
use std::io::BufWriter;
use std::path::Path;
use std::time::Instant;

use log::{error, info};
use rand::Rng;

use crate::deep::layer::{StatefulLayerWrapper, StatelessLayer};
use crate::deep::mse::mse_loss;
use crate::rl::environment::Environment;

/// A single experience tuple used for DQL training.
/// 
/// Each transition represents one step in the environment.
#[derive(Clone)]
struct Transition<A> {
    /// The state vector observed before taking an action.
    state: Vec<f32>,
    /// The action taken in the given state.
    action: A,
    /// The reward received after taking the action.
    reward: f32,
    /// The state observed after taking the action.
    next_state: Vec<f32>,
    /// Whether the episode ended after this transition.
    done: bool,
}

/// A fixed-capacity replay buffer storing past transitions for DQL.
///
/// Older transitions are removed once the buffer reaches its maximum size.
/// Sampling uses uniform random selection.
struct ReplayBuffer<A> {
    /// Maximum capacity of the buffer.
    capacity: usize,
    /// The backing buffer array.
    buffer: VecDeque<Transition<A>>,
}

impl<A: Clone> ReplayBuffer<A> {
    /// Create a new replay buffer with the given maximum capacity.
    fn new(capacity: usize) -> Self {
        Self {
            capacity,
            buffer: VecDeque::with_capacity(capacity),
        }
    }

    /// Insert a new transition, removing the oldest if capacity is exceeded.
    fn push(&mut self, transition: Transition<A>) {
        if self.buffer.len() >= self.capacity {
            self.buffer.pop_front();
        }
        self.buffer.push_back(transition);
    }

    /// Sample a batch of transitions uniformly at random.
    fn sample<'a>(&'a self, batch_size: usize) -> Vec<&'a Transition<A>> {
        let len = self.buffer.len();

        let indices = rand::seq::index::sample(&mut rand::thread_rng(), len, batch_size);

        indices
            .into_iter()
            .map(|i| &self.buffer[i])
            .collect()
    }

    /// Length of the buffer.
    fn len(&self) -> usize {
        self.buffer.len()
    }
}

/// A trainer implementing the Double Deep Q-Network (DQL) algorithm.
/// 
/// # Type Parameters
/// - `E`: The environment type, implementing the `Environment` trait.
pub struct DoubleDQLTrainer<E: Environment, T: StatelessLayer + Clone> {
    /// The Deep Q learning network to train.
    dql: StatefulLayerWrapper<T>,
    /// The target network to use in training.
    target_dql: StatefulLayerWrapper<T>,
    /// Replay buffer of seen actions.
    buffer: ReplayBuffer<E::Action>,
    /// Weight decay for future rewards.
    gamma: f32,
    /// Eps for the eps greedy algorithm.
    epsilon: f32,
    /// Eps decay for the eps greedy algorithm.
    epsilon_decay: f32,
    /// Minimum eps allowed.
    epsilon_min: f32,
    /// Learning rate for back propagation.
    learning_rate: f32,
}

impl<E: Environment, T: StatelessLayer + Clone> DoubleDQLTrainer<E, T> {
    pub fn new(dql: T, gamma: f32, epsilon: f32, epsilon_decay: f32, epsilon_min: f32, learning_rate: f32, buffer_capacity: usize) -> Self {
        let buffer = ReplayBuffer::new(buffer_capacity);

        Self {
            dql: StatefulLayerWrapper::new(dql.clone()),
            target_dql: StatefulLayerWrapper::new(dql),
            buffer,
            gamma,
            epsilon,
            epsilon_decay,
            epsilon_min,
            learning_rate,
        }
    }

    /// Select an action using eps greedy agorithm
    /// 
    /// # Arguments
    /// 
    /// * `state` - Vector representation of the state
    /// * `all_actions` - All avaliable actions in the environment. This should be fixed.
    /// * `action_mask` -  A mask of valid action at the given step.
    fn select_action(
        &mut self,
        state: &Vec<f32>,
        all_actions: &Vec<E::Action>,
        action_mask: &Vec<bool>,
    ) -> E::Action {
        let mut rng = rand::thread_rng();

        let valid_indices: Vec<usize> = action_mask
            .iter()
            .enumerate()
            .filter_map(|(i, &m)| if m { Some(i) } else { None })
            .collect();

        let has_valid_actions = !valid_indices.is_empty();

        if rng.gen_range(0.0..1.0) < self.epsilon {
            if has_valid_actions {
                let idx = valid_indices[rng.gen_range(0..valid_indices.len())];
                return all_actions[idx].clone();
            } else {
                // Fall back to selecting from all actions if none are valid
                let idx = rng.gen_range(0..all_actions.len());
                return all_actions[idx].clone();
            }
        }

        let mut q_values = self.dql.forward(state);

        if has_valid_actions {
            for (i, valid) in action_mask.iter().enumerate() {
                if !valid {
                    q_values[i] = f32::NEG_INFINITY;
                }
            }
        }

        let selected_idx = q_values
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .map(|(i, _)| i)
                .unwrap_or_else(|| {
                    error!("There should have been a valid q value. Defaulting to 0");
                    0
                });

        all_actions[selected_idx].clone()
    }

    /// Train the DQL for a fixed number of episodes.
    ///
    /// For each episode:
    /// - Reset the environment
    /// - Take up to `max_moves` actions using epsilon greedy algorithm
    /// - Store transitions in the replay buffer
    /// - Every batch_size samples, perform gradient steps
    /// - Decay epsilon between episodes
    pub fn train(&mut self, env: &mut E, num_episodes: usize, batch_size: usize, max_moves: usize, update_target_after_steps: usize, save_model_loc: Option<&Path>) {
        if let Some(loc) = save_model_loc {
            if loc.exists() {
                panic!("Cannot proceed. Already a model at {:?}", save_model_loc);
            } else if let Some(parent) = loc.parent() {
                if !parent.exists() {
                    panic!("Directory {:?} does not exist", parent);
                }
            }
        }
        let start_time = Instant::now();
        let actions = E::all_actions();
        let mut steps = 0;
        let mut total_reward = 0.0;
        for episode in 0..num_episodes {
            let mut state = env.reset().into();

            for _t in 0..max_moves {
                let action_mask = env.get_action_mask();
                let action = self.select_action(&state, &actions, &action_mask);
                let (next_state, reward, done) = env.step(&action);
                let next_state_vec = next_state.into();
                total_reward += reward;

                self.buffer.push(Transition {
                    state: state,
                    action,
                    reward,
                    next_state: next_state_vec.clone(),
                    done,
                });

                state = next_state_vec;

                if self.buffer.len() >= batch_size {
                    self.learn(batch_size);
                    steps += 1;
                    if steps % update_target_after_steps == 0 {
                        info!("Updating target");
                        self.target_dql = self.dql.clone();
                        if let Some(loc) = save_model_loc {
                            info!("Save model at {:?}", loc);
                            let tmp_path = loc.with_extension("tmp");
                            let tmp_file = File::create(&tmp_path).unwrap();
                            let writer = BufWriter::new(tmp_file);

                            serde_json::to_writer_pretty(writer, &self.dql.inner).unwrap();
                            fs::rename(tmp_path, loc).unwrap();
                        }
                        steps = 0;
                    }
                }

                if done {
                    break;
                }
            }

            if (episode + 1) % 100 == 0 {
                info!(
                    "Episode {}: average reward over last 100 episodes = {:.2} with epsilon = {:.3}",
                    episode + 1,
                    total_reward / 100.0,
                    self.epsilon
                );
                total_reward = 0.0;
            }
            self.epsilon = (self.epsilon * self.epsilon_decay).max(self.epsilon_min);
        }
        let end_time = Instant::now();
        let duration = end_time.duration_since(start_time);
        info!("Train took {duration:?}.\nParameters: num_episodes={num_episodes}, batch_size={batch_size}, max_moves={max_moves}, update_target_after_steps={update_target_after_steps}");
    }

    /// Perform one gradient-descent update on the DQL using a batch from replay.
    ///
    /// The method:
    /// - Samples transitions
    /// - Computes Q(s, a) and Q(next_s, ·)
    /// - Builds the target according to the Bellman equation
    /// - Computes loss
    /// - Backpropagates the loss through the network
    fn learn(&mut self, batch_size: usize)
    where
        E::Action: Clone,
    {
        let batch = self.buffer.sample(batch_size);

        // Prepare vectors for batch forward/backward
        let mut batch_states: Vec<&[f32]> = Vec::with_capacity(batch.len());
        let mut batch_next_states: Vec<&[f32]> = Vec::with_capacity(batch.len());
        let mut actions: Vec<usize> = Vec::with_capacity(batch.len());
        let mut rewards: Vec<f32> = Vec::with_capacity(batch.len());
        let mut dones: Vec<bool> = Vec::with_capacity(batch.len());

        for t in &batch {
            batch_states.push(&t.state);
            batch_next_states.push(&t.next_state);
            actions.push(E::action_to_index(&t.action));
            rewards.push(t.reward);
            dones.push(t.done);
        }

        // Forward pass for current states
        let batch_q_values: Vec<Vec<f32>> = self.dql.forward_batch(&batch_states);

        // Forward pass for next states
        let batch_next_q_values: Vec<Vec<f32>> = self.target_dql.forward_batch(&batch_next_states);

        // Build target vectors
        let mut batch_targets: Vec<Vec<f32>> = Vec::with_capacity(batch.len());

        for i in 0..batch.len() {
            let mut target_vec = batch_q_values[i].clone();
            let target_value = if dones[i] {
                rewards[i]
            } else {
                rewards[i]
                    + self.gamma
                        * batch_next_q_values[i]
                            .iter()
                            .copied()
                            .max_by(|a, b| a.total_cmp(b))
                            .unwrap_or_else(|| {
                                error!(
                                    "No values returned from next_q_values for sample {}. Using -inf",
                                    i
                                );
                                f32::NEG_INFINITY
                            })
            };

            if target_value.is_nan() || target_value.is_infinite() {
                // by not setting the action -> mse(target, pred)=0 -> no update
                error!("Got target {target_value} for sample {}. Not setting the target vec action", i);
            } else {
                target_vec[actions[i]] = target_value;
            }

            batch_targets.push(target_vec);
        }

        // Compute batch gradients using mse_loss
        let batch_grads: Vec<Vec<f32>> = batch_q_values.iter().zip(batch_targets.iter()).map(|(pred, target)| mse_loss(pred, target).1).collect();

        // Convert to slices for backward_batch
        let batch_grads_slices: Vec<&[f32]> = batch_grads.iter().map(|g| g.as_slice()).collect();

        // Backpropagate the batch
        self.dql.backward_batch(&batch_grads_slices, self.learning_rate);
    }
}

#[cfg(test)]
mod tests {
    use crate::deep::{linear::{StatelessLinear, WeightInit}, relu::StatelessReLU, sequential::StatelessSequential};

    use super::*;

    // ---------------------------
    // Test Environment
    // X=start, Y=goal, D=death
    // Arrows are on way only
    //
    // . -> . -> . -> . -> .
    // ^                   |
    // |                   \/
    // . <- . <- .         Y
    //      |    ^ 
    //      \/   |
    // X -> . -> . -> . -> . -> D
    //           |
    //           \/
    //           D
    // ---------------------------

    #[derive(Clone, Debug, PartialEq)]
    enum TestAction {
        Left,
        Right,
        Up,
        Down,
    }

    #[derive(Clone, Debug, PartialEq)]
    struct TestState {
        goal_loc: (usize, usize),
        player_loc: (usize, usize),
        death_loc_1: (usize, usize),
        death_loc_2: (usize, usize),
    }

    impl TestState {
        pub fn new() -> Self {
            Self { player_loc: (2, 0), goal_loc: (1, 4), death_loc_1: (2, 5), death_loc_2: (3, 2) }
        }
    }

    struct TestEnv {
        state: TestState,
    }

    impl TestEnv {
        fn new() -> Self {
            Self {
                state: TestState::new(),
            }
        }

        fn is_goal(&self, s: &TestState) -> bool {
            s.player_loc == s.goal_loc
        }

        fn is_death(&self, s: &TestState) -> bool {
            s.player_loc == s.death_loc_1 || s.player_loc == s.death_loc_2
        }

        fn get_action_mask_of_state(state: &TestState) -> Vec<bool> {
            // left, right, up, down
            let mut left = false;
            let mut right = false;
            let mut up = false;
            let mut down = false;

            match (state.player_loc.0, state.player_loc.1) {
                // Row 0
                (0, 0) => right = true,

                (0, 1) => right = true,

                (0, 2) => right = true,

                (0, 3) => right = true,

                (0, 4) => down = true,

                // Row 1
                (1, 0) => up = true,

                (1, 1) => {
                    left = true;
                    down = true;
                },

                (1, 2) => left = true,

                (1, 4) => (),

                // Row 2
                (2, 0) => right = true,

                (2, 1) => right = true,

                (2, 2) => {
                    right = true;
                    down = true;
                    up = true;
                },

                (2, 3) => right = true,

                (2, 4) => right = true,

                (2, 5) => (),

                // Row 3
                (3, 2) => (),

                // all other points should be unreachable
                _ => {
                    eprintln!("Invalid state: {:#?}", state);
                    ()
                },
            };
            vec![left, right, up, down]
        }

        fn transition(&self, state: &TestState, action: &TestAction) -> TestState {

            let mask = Self::get_action_mask_of_state(state);
            let indx = Self::action_to_index(action);
            if !mask[indx] {
                eprintln!("Invalid action for state. Action: {:#?}. State: {:#?}.", action, state);
                return state.clone()
            }

            match action {
                TestAction::Left => TestState { player_loc: (state.player_loc.0, state.player_loc.1 - 1), ..state.clone() },
                TestAction::Right => TestState { player_loc: (state.player_loc.0, state.player_loc.1 + 1), ..state.clone() },
                TestAction::Up => TestState { player_loc: (state.player_loc.0 - 1, state.player_loc.1), ..state.clone() },
                TestAction::Down => TestState { player_loc: (state.player_loc.0 + 1, state.player_loc.1), ..state.clone() },
            }
        }

        fn get_model() -> StatelessSequential {
            let mut seq = StatelessSequential::new();
            seq.add(crate::deep::sequential::SequentialLayer::Linear(StatelessLinear::new(8, 8, WeightInit::He)));
            seq.add(crate::deep::sequential::SequentialLayer::ReLU(StatelessReLU::new()));
            seq.add(crate::deep::sequential::SequentialLayer::Linear(StatelessLinear::new(8, 4, WeightInit::He)));
            seq
        }
    }

    impl Into<Vec<f32>> for TestState {
        fn into(self) -> Vec<f32> {
            vec![self.player_loc.0 as f32, self.player_loc.1 as f32, self.goal_loc.0 as f32, self.goal_loc.1 as f32, self.death_loc_1.0 as f32, self.death_loc_1.1 as f32, self.death_loc_2.0 as f32, self.death_loc_2.1 as f32]
        }
    }

    impl Environment for TestEnv {
        type State = TestState;
        type Action = TestAction;

        fn reset(&mut self) -> Self::State {
            self.state = TestState::new();
            self.state.clone()
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

    #[test]
    fn double_dql_reaches_goal() {
        let _ = env_logger::builder().is_test(true).try_init();
        let mut env = TestEnv::new();

        let gamma = 0.95;
        let epsilon = 0.99;
        let epsilon_decay = 0.999;
        let epsilon_min = 0.05;
        let learning_rate = 0.001;
        let buffer_capacity = 10_000;

        let dql = TestEnv::get_model();

        let mut trainer = DoubleDQLTrainer::<TestEnv, StatelessSequential>::new(
            dql,
            gamma,
            epsilon,
            epsilon_decay,
            epsilon_min,
            learning_rate,
            buffer_capacity,
        );

        let episodes = 2500;
        let batch_size = 10;
        let max_moves = 100;
        let update_target_after_steps = 100;

        trainer.train(&mut env, episodes, batch_size, max_moves, update_target_after_steps, None);

        trainer.epsilon = 0.0;

        let mut state = env.reset();
        let mut state_vec: Vec<f32> = state.clone().into();

        for step in 0..100 {
            let mask = env.get_action_mask();
            let action = trainer.select_action(&state_vec, &TestEnv::all_actions(), &mask);

            println!("Step {}: action = {:?}", step + 1, action);

            let (next_state, _reward, done) = env.step(&action);
            state_vec = next_state.clone().into();
            state = next_state;

            if env.is_goal(&state) {
                println!("Reached goal in {} steps", step + 1);
                return;
            }
            if env.is_death(&state) {
                panic!("Agent died at state {:?} after {} steps", state, step);
            }
            if done {
                break;
            }
        }

        panic!("Agent did not reach the goal within 100 steps. Final state: {:?}", state);
    }
}
