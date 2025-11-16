//! Module to perform the Deep Q Learning algorithm.
use log::error;
use rand::seq::SliceRandom;
use rand::Rng;

use crate::deep::mse::mse_loss;
use crate::rl::dqn_model::DQN;
use crate::rl::environment::Environment;

/// A single experience tuple used for DQN training.
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

/// A fixed-capacity replay buffer storing past transitions for DQN.
///
/// Older transitions are removed once the buffer reaches its maximum size.
/// Sampling uses uniform random selection.
struct ReplayBuffer<A> {
    /// Maximum capacity of the buffer.
    capacity: usize,
    /// The backing buffer array.
    buffer: Vec<Transition<A>>,
}

impl<A: Clone> ReplayBuffer<A> {
    /// Create a new replay buffer with the given maximum capacity.
    fn new(capacity: usize) -> Self {
        Self {
            capacity,
            buffer: Vec::with_capacity(capacity),
        }
    }

    /// Insert a new transition, removing the oldest if capacity is exceeded.
    fn push(&mut self, transition: Transition<A>) {
        if self.buffer.len() >= self.capacity {
            self.buffer.remove(0);
        }
        self.buffer.push(transition);
    }

    /// Sample a batch of transitions uniformly at random.
    fn sample(&self, batch_size: usize) -> Vec<Transition<A>> {
        let mut rng = rand::thread_rng();
        self.buffer
            .choose_multiple(&mut rng, batch_size)
            .cloned()
            .collect()
    }

    /// Length of the buffer.
    fn len(&self) -> usize {
        self.buffer.len()
    }
}

/// A trainer implementing the Deep Q-Network (DQN) algorithm.
/// 
/// # Type Parameters
/// - `E`: The environment type, implementing the `Environment` trait.
pub struct DQNTrainer<E: Environment> {
    /// The Deep Q learning network to train.
    dqn: DQN,
    /// The target network to use in training.
    target_dqn: DQN,
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

impl<E: Environment> DQNTrainer<E> {
    pub fn new(state_size: usize, num_of_actions: usize, gamma: f32, epsilon: f32, epsilon_decay: f32, epsilon_min: f32, learning_rate: f32) -> Self {
        let dqn = DQN::new(state_size, num_of_actions);
        let target_dqn = DQN::new(state_size, num_of_actions);
        let buffer = ReplayBuffer::new(10_000);

        Self {
            dqn,
            target_dqn,
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

        if rng.gen::<f32>() < self.epsilon {
            if has_valid_actions {
                let idx = valid_indices[rng.gen_range(0..valid_indices.len())];
                return all_actions[idx].clone();
            } else {
                // Fall back to selecting from all actions if none are valid
                let idx = rng.gen_range(0..all_actions.len());
                return all_actions[idx].clone();
            }
        }

        let mut q_values = self.dqn.forward(state);

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

    /// Train the DQN for a fixed number of episodes.
    ///
    /// For each episode:
    /// - Reset the environment
    /// - Take up to `max_moves` actions using epsilon greedy algorithm
    /// - Store transitions in the replay buffer
    /// - Every batch_size samples, perform gradient steps
    /// - Decay epsilon between episodes
    pub fn train(&mut self, env: &mut E, num_episodes: usize, batch_size: usize, max_moves: usize, episode_sync_freq: usize) {
        let actions = E::all_actions();
        for episode in 0..num_episodes {
            let mut state = env.reset().into();
            let mut total_reward = 0.0;

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
                }

                if done {
                    break;
                }
            }

            self.epsilon = (self.epsilon * self.epsilon_decay).max(self.epsilon_min);

            println!(
                "Episode {}: total reward = {:.2}, epsilon = {:.3}",
                episode + 1,
                total_reward,
                self.epsilon
            );

            if episode % episode_sync_freq == 0 {
                self.target_dqn = self.dqn.clone();
            }
        }
    }

    /// Perform one gradient-descent update on the DQN using a batch from replay.
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

        for t in batch {
            let q_values = self.dqn.forward(&t.state);
            let next_q_values = self.target_dqn.forward(&t.next_state);

            let action_index = E::action_to_index(&t.action);
            let target = if t.done {
                t.reward
            } else {
                t.reward + self.gamma * next_q_values.iter().cloned().fold(f32::MIN, f32::max)
            };

            let mut target_vec = q_values.clone();
            target_vec[action_index] = target;

            let loss = mse_loss(&q_values, &target_vec);

            self.dqn.backward(&loss.1, self.learning_rate);
        }
    }
}

#[cfg(test)]
mod tests {
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
        row: usize,
        col: usize,
    }

    struct TestEnv {
        state: TestState,
    }

    impl TestEnv {
        fn new() -> Self {
            Self {
                state: TestState { row: 2, col: 0 },
            }
        }

        fn is_goal(&self, s: &TestState) -> bool {
            s.row == 1 && s.col == 4
        }

        fn is_death(&self, s: &TestState) -> bool {
            (s.row == 2 && s.col == 5) || (s.row == 3 && s.col == 2)
        }

        fn get_action_mask_of_state(state: &TestState) -> Vec<bool> {
            // left, right, up, down
            let mut left = false;
            let mut right = false;
            let mut up = false;
            let mut down = false;

            match (state.row, state.col) {
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
                TestAction::Left => TestState { row: state.row, col: state.col - 1},
                TestAction::Right => TestState { row: state.row, col: state.col + 1 },
                TestAction::Up => TestState { row: state.row - 1, col: state.col },
                TestAction::Down => TestState { row: state.row + 1, col: state.col },
            }
        }
    }

    impl Into<Vec<f32>> for TestState {
        fn into(self) -> Vec<f32> {
            vec![self.row as f32, self.col as f32]
        }
    }

    impl Environment for TestEnv {
        type State = TestState;
        type Action = TestAction;

        fn reset(&mut self) -> Self::State {
            self.state = TestState { row: 2, col: 0 };
            self.state.clone()
        }

        fn step(&mut self, action: &Self::Action) -> (Self::State, f32, bool) {
            let next = self.transition(&self.state, action);
            self.state = next.clone();

            if self.is_goal(&next) {
                return (next, 100.0, true);
            }
            if self.is_death(&next) {
                return (next, -100.0, true);
            }

            (next, -1.0, false)
        }

        fn available_actions(&self) -> Vec<Self::Action> {
            vec![
                TestAction::Left,
                TestAction::Right,
                TestAction::Up,
                TestAction::Down,
            ]
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
    fn dqn_follows_optimal_path() {
        let mut env = TestEnv::new();

        // Input size = number of state features, output size = number of actions
        let mut trainer = DQNTrainer::<TestEnv>::new(2, 4, 0.9, 0.99, 0.99, 0.05, 0.01);

        let episodes = 500;
        let batch_size = 32;
        let max_moves = 40;
        let episode_sync_freq = 5;

        // Train the DQN
        trainer.train(&mut env, episodes, batch_size, max_moves, episode_sync_freq);

        // After training, set epsilon to 0 for greedy action selection
        trainer.epsilon = 0.0;

        // Reset environment
        let mut state = env.reset();
        let mut state_vec: Vec<f32> = state.clone().into();

        // Define the optimal path as a sequence of actions
        let optimal_path = vec![
            TestAction::Right,
            TestAction::Right,
            TestAction::Up,
            TestAction::Left,
            TestAction::Left,
            TestAction::Up,
            TestAction::Right,
            TestAction::Right,
            TestAction::Right,
            TestAction::Right,
            TestAction::Down,
        ];

        for expected_action in optimal_path {
            // Get action mask for current state
            let mask = env.get_action_mask();

            // Select action using the trained DQN
            let chosen = trainer.select_action(&state_vec, &TestEnv::all_actions(), &mask);
            println!("{:?}", chosen);

            assert_eq!(
                chosen, expected_action,
                "DQN chose wrong action at state {:?}. Expected {:?}, got {:?}",
                state, expected_action, chosen
            );

            // Take the action in the environment
            let (next_state, _reward, done) = env.step(&chosen);
            state_vec = next_state.clone().into();
            state = next_state;

            if done {
                break;
            }
        }

        // Check that we ended in the goal
        assert!(env.is_goal(&state), "DQN did not reach the goal at the end of the path");
    }

}
