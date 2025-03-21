// use std::collections::HashMap;

// use crate::models::snake::snake_model::{Snake, SnakeBlock};


// pub struct State {
//     board: Vec<Vec<SnakeBlock>>,
// }

// pub struct Action {
//     action: (i8, i8),
// }

// pub struct QTable {
//     table: HashMap<State, HashMap<Action, f32>>,
// }

// impl QTable {
//     pub fn new() -> Self {
//         Self { table: HashMap::new() }
//     }
// }

// pub fn q_learning(
//     game_environment: Snake,
//     num_episodes: u64,
//     max_episode_length: u64,
//     learning_rate: fn(u64) -> f32,
//     gamma: f32,
//     epsilon: fn(u64) -> f32,
//     q_table: Option<QTable>,
// ) -> QTable {
//     let q_table = q_table.unwrap_or_else(|| QTable::new());

//     q_table
// }

// // def q_learning(
// //     env: gymnasium.Env,
// //     num_episodes: int,
// //     max_episode_length: int,
// //     learning_rate: float,
// //     gamma: float,
// //     seed: int,
// //     epsilon: float = 1.0,
// // ) -> dict[str, dict[str, float]]:
// //     """
// //     Build a Q-Learning policy

// //     Args:
// //         env (Env): The Environment instance
// //         num_episodes (int): The number of episodes to build the table from
// //         max_episode_length (int): The maximum length of an episode to prevent infinite loops
// //         learning_rate (float): A hyperparameter denoting how quickly the agent "learns" reward values
// //         gamma (float): The discount rate
// //         epsilon (float): The probability with which you should select a random
// //         action instead of following a greedy policy

// //     Returns:
// //         dict[str, dict[str, float]]: A dictionary of dictionaries mapping a state and action
// //         to a specific reward value. This is what you will build in this algorithm
// //     """

// //     # Set up q-table
// //     q_table: dict[str, dict[str, float]] = {}
// //     random.seed(seed)

// //     ### YOUR CODE BELOW HERE
// //     def _init_state_actions(state_id: str, actions: list[str]):
// //         if not state_id in q_table:
// //             q_table[state_id] = {}
// //         for a in actions:
// //             if a not in q_table[state_id]:
// //                 q_table[state_id][a] = 0.0

// //     for episode in range(num_episodes):
// //         epsilon = abs(math.exp(-episode / 300) * math.cos(episode / 10))
// //         state_id, actions = reset_mdp(env, seed)
// //         _init_state_actions(state_id, actions)
// //         for _ in range(max_episode_length):
// //             if random.uniform(0.0, 1.0) < epsilon:
// //                 action = random.choice(actions)
// //             else:
// //                 best_action = None
// //                 best_value = float("-inf")
// //                 for act in actions:
// //                     if q_table[state_id][act] > best_value:
// //                         best_value = q_table[state_id][act]
// //                         best_action = act
// //                 if best_action is None:
// //                     print("This should not happen")
// //                     best_action = actions[0]
// //                 action = best_action
// //             next_state_id, reward, terminated, next_actions = do_action_mdp(action, env)
// //             _init_state_actions(next_state_id, next_actions)

// //             max_next_q_value = 0.0 if not next_actions else max([q_table[next_state_id][act] for act in next_actions])
// //             delta = reward + gamma * max_next_q_value - q_table[state_id][action]
// //             q_table[state_id][action] += learning_rate * delta

// //             state_id = next_state_id
// //             actions = next_actions

// //             if terminated:
// //                 break

// //     ### YOUR CODE ABOVE HERE

// //     return q_table

// // def run_policy(
// //     q_table: dict[str, dict[str, float]],
// //     env: gymnasium.Env,
// //     seed: int,
// //     max_policy_length: int = 25,
// // ) -> tuple[list[str], float]:
// //     """
// //     Run a policy from a built Q-Table

// //     Args:
// //         q_table (dict[str, dict[str, float]]): The built Q-Table dictionary
// //         env (gymnasium.Env): The environment in which to run the policy
// //         seed (int): The seed to use
// //         max_policy_length (int): The maximum length of the policy to run
        
// //     Returns:
// //         list[str]: The sequence of actions that the policy performed
// //         float: The sum total reward gained from the environment
// //     """
// //     actions = []  # Store the entire sequence of actions here
// //     total_reward = 0.0  # Store the total sum reward of all actions executed here

// //     ### YOUR CODE BELOW HERE

// //     state_id, actions = reset_mdp(env, seed)

// //     for _ in range(max_policy_length):
// //         if not actions: # no where to go
// //             break
// //         if state_id in q_table: # been explored before
// //             action = max(q_table[state_id].items(), key = lambda x: x[1])[0]
// //         else:
// //             # my q-learning apparently sucks
// //             action = random.choice(actions)
// //         actions.append(action)
// //         state_id, reward, terminal, actions = do_action_mdp(action, env)
// //         total_reward += reward
// //         if terminal:
// //             break

// //     ### YOUR CODE ABOVE HERE

// //     return actions, total_reward
