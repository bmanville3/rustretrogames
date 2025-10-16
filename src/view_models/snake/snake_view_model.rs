use std::{
    collections::HashMap,
    sync::mpsc::{self, Receiver, Sender},
    time::Duration,
};

use iced::keyboard::{key::Named, Key};
use log::{debug, error, warn};
use tokio::{runtime::Handle, task::JoinHandle};

use crate::{
    app::Message,
    models::snake::{
        snake_bot::SnakeBotType,
        snake_game::{PartialSnakeGame, SnakeAction, SnakeError, SnakeGame, MILLIS_BETWEEN_FRAMES},
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
    GetGame(Sender<ChannelMessage>),
    Game(SnakeGame),
    MoveRealPlayer((usize, SnakeAction)),
    TickBoard,
    Input(PartialSnakeGame),
    BotMove((usize, SnakeAction)),
    Kill,
}

#[derive(Debug)]
pub struct SnakeViewModel {
    params: SnakeParams,
    sender_to_main_loop: Sender<ChannelMessage>,
    main_handle: JoinHandle<()>,
    last_model_fetched: SnakeGame,
    real_player_1_index: Option<usize>,
    real_player_2_index: Option<usize>,
    game_over: bool,
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
        let number_of_real_players = params.number_of_real_players;
        let model = SnakeGame::new(
            params.number_of_bots,
            number_of_real_players,
            params.grid_size,
        )?;
        let mut real_player_1_index = None;
        let mut real_player_2_index = None;
        let mut bot_threads = HashMap::with_capacity(params.number_of_bots * 2);
        let (sender_to_main_loop, receiver_for_main_loop) = mpsc::channel::<ChannelMessage>();
        for s in model.get_all_players() {
            if s.is_bot {
                bot_threads.insert(
                    s.player_id,
                    Self::make_bot_thread(
                        &params.bot_type,
                        s.player_id,
                        sender_to_main_loop.clone(),
                    ),
                );
            } else if real_player_1_index.is_none() {
                real_player_1_index = Some(s.player_id);
            } else {
                real_player_2_index = Some(s.player_id);
            }
        }
        Ok(Self {
            params,
            sender_to_main_loop,
            main_handle: Self::main_loop(
                model.clone(),
                bot_threads,
                receiver_for_main_loop,
                real_player_1_index,
                real_player_2_index,
                number_of_real_players,
            ),
            last_model_fetched: model,
            real_player_1_index,
            real_player_2_index,
            game_over: false,
        })
    }

    #[must_use]
    pub fn main_loop(
        mut model: SnakeGame,
        mut bot_threads: HashMap<usize, (JoinHandle<()>, Sender<ChannelMessage>)>,
        receiver_for_main_loop: Receiver<ChannelMessage>,
        real_player_1_index: Option<usize>,
        real_player_2_index: Option<usize>,
        number_of_real_players: usize,
    ) -> JoinHandle<()> {
        tokio::spawn(async move {
            let mut errors_in_a_row = 0;
            loop {
                let message = match receiver_for_main_loop.recv() {
                    Ok(m) => {
                        errors_in_a_row = 0;
                        m
                    }
                    Err(e) => {
                        errors_in_a_row += 1;
                        error!("Error receiving message in main loop.: {:#?}", e);
                        if errors_in_a_row > 10 {
                            error!("Received {errors_in_a_row} errors in a row. Quitting");
                            break;
                        }
                        continue;
                    }
                };
                match message {
                    ChannelMessage::BotMove((pindx, action)) => {
                        let Some(bot) = model.get_all_players().get(pindx) else {
                            error!("Got pindx {pindx} but no bot found there");
                            continue;
                        };
                        if bot.is_dead()
                            || Self::game_over_inner(
                                &model,
                                real_player_1_index,
                                real_player_2_index,
                                number_of_real_players,
                            )
                        {
                            if let Some((handle, sender)) = bot_threads.remove(&pindx) {
                                Self::kill_in_time(150, handle, &sender);
                                debug!("Number of remaining bot threads: {}", bot_threads.len());
                            } else {
                                debug!("Could not get threads for bot {} to kill", pindx);
                            }
                            continue;
                        }
                        model.add_move_to_snake(pindx, action);
                    }
                    ChannelMessage::Kill => {
                        debug!("Killing main loop");
                        for (handle, sender) in bot_threads.into_values() {
                            Self::kill_in_time(150, handle, &sender);
                        }
                        break;
                    }
                    ChannelMessage::MoveRealPlayer((pindx, action)) => {
                        model.add_move_to_snake(pindx, action);
                    }
                    ChannelMessage::TickBoard => {
                        model.move_all_characters();
                        let mut bots_to_remove = Vec::new();
                        {
                            for (bot_id, (_hanlde, sender)) in &bot_threads {
                                let Some(bot) = model.get_all_players().get(*bot_id) else {
                                    error!("Got pindx {bot_id} but no bot found there");
                                    continue;
                                };
                                if bot.is_dead()
                                    || Self::game_over_inner(
                                        &model,
                                        real_player_1_index,
                                        real_player_2_index,
                                        number_of_real_players,
                                    )
                                {
                                    bots_to_remove.push(*bot_id);
                                } else {
                                    let partial = PartialSnakeGame::from_full(model.clone());
                                    match sender.send(ChannelMessage::Input(partial)) {
                                        Ok(()) => (),
                                        Err(e) => error!(
                                            "Error sending message to bot thread {bot_id}: {e}"
                                        ),
                                    }
                                }
                            }
                        }

                        for bot_id in bots_to_remove {
                            if let Some((handle, sender)) = bot_threads.remove(&bot_id) {
                                Self::kill_in_time(150, handle, &sender);
                                debug!("Number of remaining bot threads: {}", bot_threads.len());
                            } else {
                                debug!("Could not get threads for bot {} to kill", bot_id);
                            }
                            continue;
                        }
                    }
                    ChannelMessage::GetGame(sender) => {
                        if let Err(e) = sender.send(ChannelMessage::Game(model.clone())) {
                            error!("Problem sending model back: {:#?}", e);
                        }
                    }
                    unknown => {
                        error!(
                            "Unexpected channel message received in main loop: {:#?}",
                            unknown
                        );
                    }
                }
            }
        })
    }

    fn kill_main_if_alive(&mut self) {
        if self.main_handle.is_finished() {
            return;
        }
        if let Err(e) = self.sender_to_main_loop.send(ChannelMessage::Kill) {
            error!("Error sending kill message to main: {:#?}", e);
        }
        tokio::task::block_in_place(|| {
            let rt_handle = Handle::current();
            rt_handle.block_on(async {
                if let Err(e) =
                    tokio::time::timeout(Duration::from_secs(2), &mut self.main_handle).await
                {
                    error!("Error closing main loop in time. Forcing abort: {:#?}", e);
                    self.main_handle.abort();
                }
            });
        });
    }

    fn kill_in_time(millis: u64, mut handle: JoinHandle<()>, sender: &Sender<ChannelMessage>) {
        debug!("Killing a thread in {millis} millisonds or aborting");
        if let Err(e) = sender.send(ChannelMessage::Kill) {
            error!("Error sending kill message: {:#?}", e);
        }
        tokio::task::block_in_place(|| {
            let rt_handle = Handle::current();
            rt_handle.block_on(async {
                if let Err(e) =
                    tokio::time::timeout(Duration::from_millis(millis), &mut handle).await
                {
                    error!("Error closing thread in time. Forcing abort: {:#?}", e);
                    handle.abort();
                }
            });
        });
    }

    /// Makes a new bot thread responsbile for controlling the bot at `player_indx`.
    #[must_use]
    pub fn make_bot_thread(
        bot_type: &Option<SnakeBotType>,
        player_indx: usize,
        sender_to_view_model: Sender<ChannelMessage>,
    ) -> (JoinHandle<()>, Sender<ChannelMessage>) {
        debug!("New bot thread for player {player_indx}");
        let bot = if let Some(bt) = bot_type {
            bt.make_new_bot(player_indx)
        } else {
            error!(
                "Tried to spawn a bot thread with no bot type selected. Setting bot type to random"
            );
            SnakeBotType::RandomMoveBot.make_new_bot(player_indx)
        };
        let (sender_to_bot, receiver_for_bot) = mpsc::channel::<ChannelMessage>();
        let bot_handle = tokio::spawn(async move {
            let mut recv_errors_in_a_row = 0;
            loop {
                let message = match receiver_for_bot.recv() {
                    Ok(m) => {
                        recv_errors_in_a_row = 0;
                        m
                    }
                    Err(e) => {
                        recv_errors_in_a_row += 1;
                        error!("Error getting message: {:#?}", e);
                        continue;
                    }
                };
                match message {
                    ChannelMessage::Input(snake_game) => {
                        if let Err(e) = sender_to_view_model.send(ChannelMessage::BotMove((
                            bot.get_player_index(),
                            bot.make_move(snake_game),
                        ))) {
                            error!("Problem sending BotMove message: {}", e);
                        }
                    }
                    ChannelMessage::Kill => {
                        debug!("Bot thread {player_indx} received kill");
                        break;
                    }
                    other => {
                        error!("Received unkown message in bot thread: {:#?}", other);
                    }
                }
                if recv_errors_in_a_row > 10 {
                    error!("Bot {player_indx} could not receive 10 times in a row. Quitting");
                    return;
                }
            }
        });
        (bot_handle, sender_to_bot)
    }

    fn get_game_copy(&self) -> Option<SnakeGame> {
        let (sender, receiver) = mpsc::channel::<ChannelMessage>();
        if let Err(e) = self
            .sender_to_main_loop
            .send(ChannelMessage::GetGame(sender))
        {
            error!("Problem sending to main loop: {:#?}", e);
        }
        match receiver.recv() {
            Ok(input) => match input {
                ChannelMessage::Game(snake_game) => Some(snake_game),
                unknown => {
                    error!("Got unexpected message from get_game_copy: {:#?}", unknown);
                    None
                }
            },
            Err(e) => {
                error!("Problem getting game board: {:#?}", e);
                None
            }
        }
    }

    #[must_use]
    pub fn get_params(&self) -> SnakeParams {
        self.params.clone()
    }

    #[must_use]
    pub fn get_last_game_board_ref(&self) -> &SnakeGame {
        &self.last_model_fetched
    }

    #[must_use]
    pub fn get_number_of_players(&self) -> usize {
        self.params.number_of_bots + self.params.number_of_real_players
    }

    #[must_use]
    pub fn get_time_between_frames(&self) -> u64 {
        MILLIS_BETWEEN_FRAMES
    }

    #[must_use]
    pub fn real_players_lost_inner(
        model: &SnakeGame,
        rp1_indx: Option<usize>,
        rp2_indx: Option<usize>,
        number_of_real_players: usize,
    ) -> bool {
        // either no real players or only real players -> real players never lose
        if number_of_real_players == 0 || number_of_real_players == model.get_all_players().len() {
            return false;
        }
        let sp1_opt = if let Some(indx1) = rp1_indx {
            model.get_player(indx1)
        } else {
            error!("Tried to get the first real player index but wasn't present");
            return false;
        };
        if let Some(sp1) = sp1_opt {
            if number_of_real_players == 1 {
                return sp1.is_dead();
            }
            let sp2_opt = if let Some(indx2) = rp2_indx {
                model.get_player(indx2)
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
            false
        }
    }

    #[must_use]
    pub fn real_players_lost(&self) -> bool {
        Self::real_players_lost_inner(
            &self.last_model_fetched,
            self.real_player_1_index,
            self.real_player_2_index,
            self.get_number_of_players(),
        )
    }

    fn game_over_inner(
        model: &SnakeGame,
        rp1_indx: Option<usize>,
        rp2_indx: Option<usize>,
        number_of_real_players: usize,
    ) -> bool {
        Self::real_players_lost_inner(model, rp1_indx, rp2_indx, number_of_real_players)
            || model.get_winner().is_some()
    }

    #[must_use]
    pub fn game_over(&self) -> bool {
        if self.game_over {
            true
        } else {
            Self::game_over_inner(
                &self.last_model_fetched,
                self.real_player_1_index,
                self.real_player_2_index,
                self.params.number_of_real_players,
            )
        }
    }
}

impl ViewModel for SnakeViewModel {
    fn update(&mut self, message: Message) -> Option<Message> {
        // check if game over is cached to true
        if let Message::Snake(snake_message) = message {
            if let SnakeMessage::SnakeGameMessage(snake_game_message) = snake_message {
                match snake_game_message {
                    SnakeGameMessage::Key(key) => {
                        if self.game_over {
                            if matches!(key, Key::Named(Named::Space)) {
                                return Some(Message::Snake(SnakeMessage::SnakeGameMessage(
                                    SnakeGameMessage::Reset,
                                )));
                            }
                            return None;
                        }
                        if self.params.number_of_real_players == 0 {
                            return None;
                        }
                        let Some(rp1i) = self.real_player_1_index else {
                            error!("Checked and there were no real players but real player 1 index was empty");
                            return None;
                        };
                        let orpi = match self.real_player_2_index {
                            Some(i) => i,
                            None => rp1i,
                        };
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
                            if let Err(e) = self
                                .sender_to_main_loop
                                .send(ChannelMessage::MoveRealPlayer((pindx, action)))
                            {
                                error!("Error sending move real player: {:#?}", e);
                            }
                        }
                        None
                    }
                    SnakeGameMessage::Timer(_) => {
                        if self.game_over {
                            return None;
                        }
                        if let Err(e) = self.sender_to_main_loop.send(ChannelMessage::TickBoard) {
                            error!("Error sending tick board: {:#?}", e);
                        }
                        self.last_model_fetched = if let Some(new_game) = self.get_game_copy() {
                            new_game
                        } else {
                            self.last_model_fetched.clone()
                        };
                        self.game_over = self.game_over();
                        None
                    }
                    SnakeGameMessage::Reset => {
                        debug!("Reset requested. Cleaning up board...");
                        self.kill_main_if_alive();
                        Some(Message::Snake(SnakeMessage::SnakeGameScreenTransition(
                            self.params.clone(),
                        )))
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

impl Drop for SnakeViewModel {
    fn drop(&mut self) {
        self.kill_main_if_alive();
    }
}
