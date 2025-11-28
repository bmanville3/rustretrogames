use std::{
    collections::HashMap, sync::mpsc::{self, Receiver, Sender}, thread::{self, JoinHandle}, time::{Duration, Instant}
};

use iced::{keyboard::{Key, key::Named}};
use log::{debug, error, warn};
use rayon::prelude::*;

use crate::{
    app::Message,
    models::snake::{
        snake_bot::{SnakeBot, SnakeBotType},
        snake_game::{MILLIS_BETWEEN_FRAMES, SnakeAction, SnakeError, SnakeGame}, snake_player::SnakePlayer,
    },
    view_model::ViewModel,
    views::snake::{snake_game_screen::SnakeGameMessage, snake_mediator::SnakeMessage},
};

const TICK_WAIT_TIME_BEFORE_START: usize = 10;

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
    TickBoard,
    Input(SnakeGame),
    Moves(HashMap<usize, SnakeAction>),
    Kill(Vec<usize>),
    KillAll,
}

#[derive(Debug)]
pub struct SnakeViewModel {
    params: SnakeParams,
    sender_to_main_loop: Sender<ChannelMessage>,
    main_handle: Option<JoinHandle<()>>,
    last_model_fetched: SnakeGame,
    real_player_1_index: Option<usize>,
    real_player_2_index: Option<usize>,
    ticks_seen: usize,
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
        // make the backing game
        let number_of_real_players = params.number_of_real_players;
        let total_players = params.number_of_bots + number_of_real_players;
        let model = SnakeGame::new(
            total_players,
            params.grid_size,
        )?;
        
        // assign the first entries to real players
        let mut real_player_1_index = None;
        let mut real_player_2_index = None;
        if number_of_real_players >= 1 {
            real_player_1_index = Some(0);
        }
        if number_of_real_players >= 2 {
            real_player_2_index = Some(1);
        }

        // assign the rest of the entries to boths
        let (sender_to_main_loop, receiver_for_main_loop) = mpsc::channel::<ChannelMessage>();
        let bot_type = params.bot_type.clone().unwrap_or_else(|| {
            SnakeBotType::RandomMoveBot
        });
        let indx_to_bot_type= (number_of_real_players..total_players).map(|i| (i, bot_type.clone())).collect();
        let (bot_thread_handle, bot_thread_sender) = Self::make_bot_thread(indx_to_bot_type, sender_to_main_loop.clone());
        Ok(Self {
            params,
            sender_to_main_loop,
            main_handle: Some(Self::main_loop(
                model.clone(),
                bot_thread_handle,
                bot_thread_sender,
                receiver_for_main_loop,
                real_player_1_index,
                real_player_2_index,
                number_of_real_players,
            )),
            last_model_fetched: model,
            real_player_1_index,
            real_player_2_index,
            game_over: false,
            ticks_seen: 0
        })
    }

    #[must_use]
    pub fn main_loop(
        mut model: SnakeGame,
        bot_thread_handle: JoinHandle<()>,
        bot_thread_sender: Sender<ChannelMessage>,
        receiver_for_main_loop: Receiver<ChannelMessage>,
        real_player_1_index: Option<usize>,
        real_player_2_index: Option<usize>,
        number_of_real_players: usize,
    ) -> JoinHandle<()> {
        struct BotInfo {
            last_move_time: Instant,
            last_board_sent: Instant,
        }

        impl BotInfo {
            fn made_a_move(&self) -> bool {
                self.last_move_time > self.last_board_sent
            }

            fn reaction_time(&self) -> Duration {
                self.last_move_time.duration_since(self.last_board_sent)
            }
        }
    
        thread::spawn(move || {
            let mut bot_move_times: HashMap<usize, BotInfo> = (number_of_real_players..model.get_number_of_players())
                .map(|i| (i, BotInfo {last_board_sent: Instant::now(), last_move_time: Instant::now() }))
                .collect();
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
                            return;
                        }
                        continue;
                    }
                };
                match message {
                    ChannelMessage::Moves(moves) => {
                        let now = Instant::now();
                        for (pindx, action) in moves {
                            let Some(player) = model.get_all_players().get(pindx) else {
                                error!("Got pindx {pindx} but no bot found there");
                                continue;
                            };
                            let is_real = real_player_1_index.is_some_and(|i| i == player.player_id) || real_player_2_index.is_some_and(|i| i == player.player_id);
                            if player.is_dead() {
                                if is_real {
                                    error!("Tried to move a real player but they were dead")
                                } else {
                                    if let Err(e) = bot_thread_sender.send(ChannelMessage::Kill(vec![pindx])) {
                                        error!("Error killing bots: {:#?}", e)
                                    }
                                }
                            } else {
                                model.add_move_to_snake(pindx, action);
                                if !is_real {
                                    if let Some(bot_info) = bot_move_times.get_mut(&pindx) {
                                        bot_info.last_move_time = now;
                                        // uncomment this statement for in depth logs
                                        // its commented out because it polutes the logs very badly
                                        // debug!("Bot {pindx} moved in {:?}", bot_info.reaction_time());
                                    }
                                }
                            }
                        }
                    }
                    ChannelMessage::Kill(_) | ChannelMessage::KillAll => {
                        // There are more oppurtinies to kill all the threads, but we will only do it here to prevent bugs
                        debug!("Killing main loop");
                        Self::kill_thread(bot_thread_sender, bot_thread_handle);
                        return;
                    }
                    ChannelMessage::TickBoard => {
                        for (bot_id, bot_info) in &bot_move_times {
                            if !bot_info.made_a_move() {
                                error!("Bot {bot_id} did not make a move since receiving the board. Took {:?}", bot_info.reaction_time());
                            }
                        }

                        let all_alive_before: Vec<usize> = Self::get_all_alive_bots(&model, real_player_1_index, real_player_2_index).iter().map(|p| p.player_id).collect();
                        model.move_all_characters();
                        let all_alive_after: Vec<usize>  = Self::get_all_alive_bots(&model, real_player_1_index, real_player_2_index).iter().map(|p| p.player_id).collect();
                        let mut bots_to_remove = Vec::new();
                        for i in all_alive_before {
                            if !all_alive_after.contains(&i) {
                                bots_to_remove.push(i);
                            }
                        }
                        if !bots_to_remove.is_empty() {
                            for b in &bots_to_remove {
                                bot_move_times.remove(b);
                            }
                            if let Err(e) = bot_thread_sender.send(ChannelMessage::Kill(bots_to_remove)) {
                                error!("Error killing bots: {:#?}", e);
                            }
                        }
                        if let Err(e) = bot_thread_sender.send(ChannelMessage::Input(model.clone())) {
                            error!("Error sending board: {:#?}", e);
                        }

                        for bot_info in bot_move_times.values_mut() {
                            bot_info.last_board_sent = Instant::now();
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

    fn kill_thread(sender: Sender<ChannelMessage>, handle: JoinHandle<()>) {
        if let Err(e) = sender.send(ChannelMessage::KillAll) {
            error!("Error killing bot thread: {:#?}", e);
        }
        if let Err(e) = handle.join() {
            error!("Error joining handle: {:#?}", e);
        }
        return;
    }

    /// Makes a new bot thread responsbile for controlling the bot at `player_indx`.
    #[must_use]
    pub fn make_bot_thread(
        indx_to_bot_type: HashMap<usize, SnakeBotType>,
        sender_to_main_loop: Sender<ChannelMessage>,
    ) -> (JoinHandle<()>, Sender<ChannelMessage>) {
        let (sender_to_bot, receiver_for_bot) = mpsc::channel::<ChannelMessage>();
        let bot_handle = thread::spawn(move || {
            let mut bots: HashMap<usize, Box<dyn SnakeBot>> = indx_to_bot_type.into_iter().map(|(i, bt)| (i, bt.make_new_bot(i))).collect();
            bots.iter_mut().for_each(|(_, b)| b.warmup());
            let mut recv_errors_in_a_row = 0;
            loop {
                let message = match receiver_for_bot.recv() {
                    Ok(m) => {
                        recv_errors_in_a_row = 0;
                        m
                    }
                    Err(e) => {
                        recv_errors_in_a_row += 1;
                        if recv_errors_in_a_row > 10 {
                            error!("Bot thread could not receive 10 times in a row. Quitting");
                            return;
                        }
                        error!("Error getting message: {:#?}", e);
                        continue;
                    }
                };
                match message {
                    ChannelMessage::Input(snake_game) => {
                        let moves = bots
                            .par_iter()
                            .map(|(i, b)| (*i, b.make_move(&snake_game)))
                            .collect();

                        if let Err(e) = sender_to_main_loop.send(ChannelMessage::Moves(moves)) {
                            error!("Error sending message: {:#?}", e);
                        }
                    }
                    ChannelMessage::Kill(indices) => {
                        debug!("Bot thread received kill {:#?}", indices);
                        for i in indices {
                            bots.remove(&i);
                        }
                    }
                    ChannelMessage::KillAll => {
                        return;
                    }
                    other => {
                        error!("Received unkown message in bot thread: {:#?}", other);
                    }
                }
            }
        });
        (bot_handle, sender_to_bot)
    }

    fn get_all_alive_bots(game: &SnakeGame, real_player_1_index: Option<usize>, real_player_2_index: Option<usize>) -> Vec<&SnakePlayer> {
        game.get_all_players().iter().filter(|p| p.is_alive()
                    && !(
                        real_player_1_index.is_some_and(|i| i == p.player_id) ||
                        real_player_2_index.is_some_and(|i| i == p.player_id))
                    ).collect()
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

    pub fn get_real_player_indices(&self) -> Vec<usize> {
        let mut indices = Vec::new();
        if let Some(rp1i) = self.real_player_1_index {
            indices.push(rp1i);
        }
        if let Some(rp2i) = self.real_player_2_index {
            indices.push(rp2i);
        }
        indices
    }

    #[must_use]
    pub fn real_players_lost_inner(
        model: &SnakeGame,
        rp1_indx: Option<usize>,
        rp2_indx: Option<usize>,
        number_of_real_players: usize,
    ) -> bool {
        // either no real players or only real players -> real players never lose
        if number_of_real_players == 0 || number_of_real_players == model.get_number_of_players() {
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
                        if self.ticks_seen < TICK_WAIT_TIME_BEFORE_START {
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
                            let mut moves = HashMap::new();
                            moves.insert(pindx, action);
                            if let Err(e) = self
                                .sender_to_main_loop
                                .send(ChannelMessage::Moves(moves))
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
                        self.ticks_seen += 1;
                        if self.ticks_seen < TICK_WAIT_TIME_BEFORE_START {
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
                        if let Some(handle) = self.main_handle.take() {
                            SnakeViewModel::kill_thread(self.sender_to_main_loop.clone(), handle);
                        }
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
        if let Some(handle) = self.main_handle.take() {
            SnakeViewModel::kill_thread(self.sender_to_main_loop.clone(), handle);
        }
    }
}
