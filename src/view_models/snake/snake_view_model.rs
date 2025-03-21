use std::{
    cmp::max,
    thread,
    time::{Duration, Instant},
};

use iced::{
    futures::{
        channel::mpsc::{self, Sender},
        SinkExt, Stream, StreamExt,
    },
    keyboard::{key::Named, Key},
    stream,
};
use log::{debug, error, warn};

use crate::{
    app::Message,
    models::snake::{
        snake_bot::{SnakeBot, SnakeBotType},
        snake_game::{self, SnakeAction, SnakeError, SnakeGame},
        snake_player::SnakePlayer,
    },
    view_model::ViewModel,
    views::snake::{snake_game_screen::SnakeGameMessage, snake_mediator::SnakeMessage},
};

#[derive(Clone, Debug)]
pub struct SnakeParams {
    pub bot_type: Option<SnakeBotType>,
    pub grid_size: usize,
    pub number_of_real_players: usize,
    pub number_of_bots: usize,
}

#[derive(Clone, Debug)]
pub enum ChannelMessage {
    BotReady((usize, mpsc::Sender<ChannelMessage>)),
    Input(SnakeGame),
    Idle(u64),
    BotMove((usize, SnakeAction, mpsc::Sender<ChannelMessage>)),
    Kill,
}

#[derive(Debug)]
pub struct SnakeViewModel {
    model: SnakeGame,
    params: SnakeParams,
    real_player_1_index: Option<usize>,
    real_player_2_index: Option<usize>,
}

impl SnakeViewModel {
    /// Creates a new view model with the given paramters.
    ///
    /// # Errors
    ///
    /// If the new parameters are invalid and a [`SnakeGame`] cannot
    /// be created, returns a [`SnakeError`].
    pub fn new(params: SnakeParams) -> Result<Self, SnakeError> {
        debug!("New SnakeViewModel params: {:#?}", params);
        let model = SnakeGame::new(
            params.number_of_bots,
            params.number_of_real_players,
            params.grid_size,
        )?;
        let mut real_player_1_index = None;
        let mut real_player_2_index = None;
        for s in model.get_all_players() {
            if !s.is_bot {
                if real_player_1_index.is_none() {
                    real_player_1_index = Some(s.player_id);
                } else {
                    real_player_2_index = Some(s.player_id);
                    break;
                }
            }
        }
        Ok(Self {
            model,
            params,
            real_player_1_index,
            real_player_2_index,
        })
    }

    // not the most scalable but number of types will remain small
    #[must_use]
    pub fn get_all_bot_types() -> [SnakeBotType; 1] {
        SnakeBotType::VALUES
    }

    /// Makes a new bot thread responsbile for controlling the bot at `player_indx`.
    pub fn make_bot_thread(&self, player_indx: usize) -> impl Stream<Item = Message> {
        let bot = if let Some(bt) = &self.params.bot_type {
            bt.make_new_bot(player_indx)
        } else {
            error!(
                "Tried to spawn a bot thread with no bot type selected. Setting bot type to random"
            );
            SnakeBotType::RandomMoveBot.make_new_bot(player_indx)
        };
        stream::channel(100, |mut output| async move {
            let (sender, mut receiver) = mpsc::channel::<ChannelMessage>(100);
            match output
                .send(Message::Snake(SnakeMessage::SnakeGameMessage(
                    SnakeGameMessage::ChannelMessage(ChannelMessage::BotReady((
                        bot.get_player_index(),
                        sender.clone(),
                    ))),
                )))
                .await
            {
                Ok(()) => (),
                Err(e) => {
                    error!("Problem sending BotReady message: {}", e);
                    return;
                }
            }
            loop {
                let input = receiver.next().await;
                if let Some(message) = input {
                    match message {
                        ChannelMessage::Input(snake_game) => {
                            match output
                                .send(Message::Snake(SnakeMessage::SnakeGameMessage(
                                    SnakeGameMessage::ChannelMessage(ChannelMessage::BotMove((
                                        bot.get_player_index(),
                                        bot.make_move(snake_game),
                                        sender.clone(),
                                    ))),
                                )))
                                .await
                            {
                                Ok(()) => (),
                                Err(e) => {
                                    debug!("Problem sending BotMove message: {}", e);
                                }
                            }
                        }
                        ChannelMessage::Idle(millis) => {
                            thread::sleep(Duration::from_millis(millis));
                        }
                        ChannelMessage::Kill => break,
                        _ => (),
                    }
                    match output
                        .send(Message::Snake(SnakeMessage::SnakeGameMessage(
                            SnakeGameMessage::ChannelMessage(ChannelMessage::BotReady((
                                bot.get_player_index(),
                                sender.clone(),
                            ))),
                        )))
                        .await
                    {
                        Ok(()) => (),
                        Err(e) => {
                            debug!("Problem sending BotReady message: {}", e);
                        }
                    }
                } else {
                    debug!("Got None in receiver. Breaking...");
                    break;
                }
            }
        })
    }

    fn kill_bot(pindx: usize, mut sender: Sender<ChannelMessage>) {
        debug!("Killing bot {}", pindx);
        tokio::spawn(async move {
            if let Err(e) = sender.send(ChannelMessage::Kill).await {
                debug!("Problem sending kill message to {pindx}: {e}");
            }
        });
    }

    fn check_bot_ready(&self, pindx: usize, mut sender: Sender<ChannelMessage>) -> bool {
        if self.params.bot_type.is_none() {
            warn!("Tried to check if bot is ready when no bot type is selected");
            return false;
        }
        let Some(player) = self.model.get_player(pindx) else {
            error!("Player at index {} not found", pindx);
            return false;
        };
        if self.game_over() || player.is_dead() {
            Self::kill_bot(pindx, sender.clone());
            return false;
        }

        let time_between_moves = u128::from(max(
            snake_game::PLAYER_MOVEMENT_CAP,
            self.params.bot_type.as_ref().unwrap().get_move_time(),
        ));

        if !player.is_bot {
            error!(
                "Player at index {} was expected to be a bot. Player name: {}",
                pindx,
                player.get_name()
            );
            return false;
        }

        let millis_diff = Instant::now()
            .duration_since(player.time_of_last_action)
            .as_millis();

        let max_wait = time_between_moves.saturating_sub(millis_diff);

        if max_wait == 0 {
            return true;
        }
        // have the bot idle for max_wait time
        tokio::spawn(async move {
            match max_wait.try_into() {
                Ok(wait) => {
                    if let Err(e) = sender.send(ChannelMessage::Idle(wait)).await {
                        debug!("Problem sending idle message: {e}");
                    }
                }
                Err(e) => {
                    warn!(
                        "Problem converting u128 to u64: {}. Sending default wait",
                        e
                    );
                    if let Err(e) = sender
                        .send(ChannelMessage::Idle(snake_game::MILLIS_BETWEEN_FRAMES))
                        .await
                    {
                        debug!("Problem sending idle message: {e}");
                    }
                }
            }
        });
        false
    }

    fn handle_channel_message(&mut self, message: ChannelMessage) {
        match message {
            ChannelMessage::BotReady((pindx, sender)) => {
                if self.check_bot_ready(pindx, sender.clone()) {
                    let mut csend = sender;
                    let model = self.model.clone();
                    tokio::spawn(async move {
                        if let Err(e) = csend.send(ChannelMessage::Input(model)).await {
                            debug!("Problem sending input message: {e}");
                        }
                    });
                }
            }
            ChannelMessage::BotMove((pindx, action, sender)) => {
                if self.check_bot_ready(pindx, sender.clone())
                    && !self
                        .model
                        .move_character(pindx, action, Some(Instant::now()), false)
                {
                    Self::kill_bot(pindx, sender);
                }
            }
            cm => {
                debug!("Ignoring unexpected channel message: {:#?}", cm);
            }
        }
    }

    #[must_use]
    pub fn get_game_ref(&self) -> &SnakeGame {
        &self.model
    }

    #[must_use]
    pub fn get_winner_indx(&self) -> Option<usize> {
        self.model.get_winner()
    }

    #[must_use]
    pub fn get_winner(&self) -> Option<&SnakePlayer> {
        match self.get_winner_indx() {
            Some(wi) => self.model.get_player(wi),
            None => None,
        }
    }

    #[must_use]
    pub fn get_time_between_frames(&self) -> u64 {
        snake_game::MILLIS_BETWEEN_FRAMES
    }

    #[must_use]
    pub fn get_number_of_players(&self) -> usize {
        self.model.get_all_players().len()
    }

    #[must_use]
    pub fn get_players(&self) -> &Vec<SnakePlayer> {
        self.model.get_all_players()
    }

    #[must_use]
    pub fn get_params(&self) -> SnakeParams {
        self.params.clone()
    }

    #[must_use]
    pub fn real_players_lost(&self) -> bool {
        // either no real players or only real players -> real players never lose
        if self.params.number_of_real_players == 0
            || self.params.number_of_real_players == self.get_number_of_players()
        {
            return false;
        }
        let sp1_opt = if let Some(indx1) = self.real_player_1_index {
            self.model.get_player(indx1)
        } else {
            error!("Tried to get the first real player index but wasn't present");
            return false;
        };
        if let Some(sp1) = sp1_opt {
            if self.params.number_of_real_players == 1 {
                return sp1.is_dead();
            }
            let sp2_opt = if let Some(indx1) = self.real_player_1_index {
                self.model.get_player(indx1)
            } else {
                error!("Tried to get the second real player index but wasn't present");
                return sp1.is_dead();
            };
            if let Some(sp2) = sp2_opt {
                sp1.is_dead() && sp2.is_dead()
            } else {
                error!("Second real player was None");
                sp1.is_dead()
            }
        } else {
            error!("First real player was None");
            true
        }
    }

    #[must_use]
    pub fn game_over(&self) -> bool {
        self.real_players_lost() || self.get_winner().is_some()
    }
}

impl ViewModel for SnakeViewModel {
    fn update(&mut self, message: Message) -> Option<Message> {
        if let Message::Snake(snake_message) = message {
            if let SnakeMessage::SnakeGameMessage(snake_game_message) = snake_message {
                match snake_game_message {
                    SnakeGameMessage::ChannelMessage(channel_message) => {
                        self.handle_channel_message(channel_message);
                        None
                    }
                    SnakeGameMessage::Key(key) => {
                        if self.game_over() {
                            if matches!(key, Key::Named(Named::Space)) {
                                return Some(Message::Snake(SnakeMessage::SnakeGameMessage(
                                    SnakeGameMessage::Reset(true),
                                )));
                            }
                            return None;
                        }
                        if self.params.number_of_real_players == 0 {
                            return None;
                        }
                        let rp1i = self.real_player_1_index.unwrap();
                        let orpi = if self.real_player_2_index.is_some() {
                            self.real_player_2_index.unwrap()
                        } else {
                            rp1i
                        };
                        // TODO: Check the time of both players
                        let movement = match key {
                            Key::Named(code) => match code {
                                Named::ArrowUp => Some((rp1i, SnakeAction::Up)),
                                Named::ArrowDown => Some((rp1i, SnakeAction::Down)),
                                Named::ArrowLeft => Some((rp1i, SnakeAction::Left)),
                                Named::ArrowRight => Some((rp1i, SnakeAction::Right)),
                                _ => None,
                            },
                            Key::Character(ref c) => match c.as_str() {
                                "w" | "W" => Some((orpi, SnakeAction::Up)),
                                "s" | "S" => Some((orpi, SnakeAction::Down)),
                                "a" | "A" => Some((orpi, SnakeAction::Left)),
                                "d" | "D" => Some((orpi, SnakeAction::Right)),
                                _ => None,
                            },
                            Key::Unidentified => None,
                        };

                        if let Some((pindx, action)) = movement {
                            self.model
                                .move_character(pindx, action, Some(Instant::now()), false);
                        }
                        None
                    }
                    SnakeGameMessage::Timer(_) => {
                        if self.game_over() {
                            return None;
                        }
                        let pids_and_last_actions: Vec<(usize, SnakeAction)> = self
                            .model
                            .get_all_players()
                            .iter()
                            .filter(|p| p.is_alive())
                            .map(|p| (p.player_id, p.last_action.clone()))
                            .collect();
                        for (pid, last_action) in pids_and_last_actions {
                            self.model.move_character(pid, last_action, None, true);
                        }
                        None
                    }
                    SnakeGameMessage::Reset(reset) => {
                        if reset {
                            Some(Message::Snake(SnakeMessage::SnakeGameScreenTransition(
                                self.params.clone(),
                            )))
                        } else {
                            None
                        }
                    }
                }
            } else {
                warn!(
                    "Non-SnakeGameMessage sent to SnakeViewModel: {:#?}",
                    snake_message
                );
                None
            }
        } else {
            warn!("Non-snake message sent to SnakeViewModel: {:#?}", message);
            None
        }
    }
}
