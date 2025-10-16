// use rand::seq::SliceRandom;
// use rand::Rng;

// use crate::rl::dqn_model::DQN;
// use crate::rl::environment::Environment;

// #[derive(Clone)]
// struct Transition<A> {
//     state: Vec<f32>,
//     action: A,
//     reward: f32,
//     next_state: Vec<f32>,
//     done: bool,
// }

// struct ReplayBuffer<A> {
//     capacity: usize,
//     buffer: Vec<Transition<A>>,
// }

// impl<A: Clone> ReplayBuffer<A> {
//     fn new(capacity: usize) -> Self {
//         Self {
//             capacity,
//             buffer: Vec::with_capacity(capacity),
//         }
//     }

//     fn push(&mut self, transition: Transition<A>) {
//         if self.buffer.len() >= self.capacity {
//             self.buffer.remove(0);
//         }
//         self.buffer.push(transition);
//     }

//     fn sample(&self, batch_size: usize) -> Vec<Transition<A>> {
//         let mut rng = rand::thread_rng();
//         self.buffer
//             .choose_multiple(&mut rng, batch_size)
//             .cloned()
//             .collect()
//     }

//     fn len(&self) -> usize {
//         self.buffer.len()
//     }
// }

// /// Trainer for a DQN model
// pub struct DQNTrainer<E: Environment> {
//     dqn: DQN,
//     target_dqn: DQN,
//     buffer: ReplayBuffer<E::Action>,
//     gamma: f32,
//     epsilon: f32,
//     epsilon_decay: f32,
//     epsilon_min: f32,
//     learning_rate: f32,
// }

// impl<E: Environment> DQNTrainer<E> {
//     pub fn new(input_dim: usize, output_dim: usize) -> Self {
//         let dqn = DQN::new(input_dim, output_dim);
//         let target_dqn = DQN::new(input_dim, output_dim);
//         let buffer = ReplayBuffer::new(10_000);

//         Self {
//             dqn,
//             target_dqn,
//             buffer,
//             gamma: 0.99,
//             epsilon: 1.0,
//             epsilon_decay: 0.995,
//             epsilon_min: 0.05,
//             learning_rate: 0.001,
//         }
//     }

//     /// Select action via eps-greedy policy
//     fn select_action(&mut self, state: &Vec<f32>, actions: &[E::Action]) -> E::Action {
//         let mut rng = rand::thread_rng();
//         if rng.gen::<f32>() < self.epsilon {
//             actions.choose(&mut rng).unwrap().clone()
//         } else {
//             let q_values = self.dqn.forward(state.clone());
//             let max_idx = q_values
//                 .iter()
//                 .enumerate()
//                 .max_by(|a, b| a.1.total_cmp(b.1))
//                 .map(|(i, _)| i)
//                 .unwrap_or(0);
//             actions[max_idx % actions.len()].clone()
//         }
//     }

//     /// Train the DQN on a given environment for N episodes
//     pub fn train(&mut self, env: &mut E, num_episodes: usize, batch_size: usize, max_moves: usize) {
//         for episode in 0..num_episodes {
//             let mut state = env.reset();
//             let mut total_reward = 0.0;

//             for _t in 0..max_moves {
//                 let actions = env.available_actions();
//                 let action = self.select_action(&state_to_vec(&state), &actions);
//                 let (next_state, reward, done) = env.step(&action);
//                 total_reward += reward;

//                 self.buffer.push(Transition {
//                     state: state_to_vec(&state),
//                     action: action.clone(),
//                     reward,
//                     next_state: state_to_vec(&next_state),
//                     done,
//                 });

//                 state = next_state;

//                 if self.buffer.len() >= batch_size {
//                     self.learn(batch_size);
//                 }

//                 if done {
//                     break;
//                 }
//             }

//             // Decay epsilon
//             self.epsilon = (self.epsilon * self.epsilon_decay).max(self.epsilon_min);

//             println!(
//                 "Episode {}: total reward = {:.2}, epsilon = {:.3}",
//                 episode + 1,
//                 total_reward,
//                 self.epsilon
//             );
//         }
//     }

//     /// Perform one learning step (gradient descent)
//     fn learn(&mut self, batch_size: usize)
//     where
//         E::Action: Clone,
//     {
//         let batch = self.buffer.sample(batch_size);

//         for t in batch {
//             let q_values = self.dqn.forward(t.state.clone());
//             let next_q_values = self.target_dqn.forward(t.next_state.clone());

//             // assume action index corresponds to vector position
//             let action_index = 0; // you'd map actions to indices in a real env
//             let target = if t.done {
//                 t.reward
//             } else {
//                 t.reward + self.gamma * next_q_values.iter().cloned().fold(f32::MIN, f32::max)
//             };

//             let mut target_vec = q_values.clone();
//             target_vec[action_index] = target;

//             let loss = mse_loss(&q_values, &target_vec);

//             // Backpropagate manually using your small NN module
//             self.dqn.net.backward(loss);
//             self.dqn.net.step(self.learning_rate);
//         }
//     }
// }

// /// Convert state type to a flat vector
// fn state_to_vec<S>(state: &S) -> Vec<f32>
// where
//     S: Clone + Into<Vec<f32>>,
// {
//     state.clone().into()
// }
