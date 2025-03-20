use std::{
    thread,
    time::{Duration, Instant},
};

use iced::{
    futures::{channel::mpsc, SinkExt, Stream, StreamExt},
    keyboard::{key::Named, Key},
    stream,
};
use log::{debug, warn};
use rand::Rng;

use crate::{
    app::Message,
    models::snake::{
        snake_bot::{make_new_bot, SnakeBot, SnakeBotType},
        snake_model::{self, Snake, SnakeBlock},
    },
    view_model::ViewModel,
    views::snake::{snake_game_screen::SnakeGameMessage, snake_mediator::SnakeMessage},
};

#[derive(Clone, Debug)]
pub enum ChannelMessage {
    BotReady(mpsc::Sender<ChannelMessage>),
    Input(Vec<Vec<SnakeBlock>>),
    Idle(u64),
    BotMove((i8, i8)),
    Kill,
}

#[derive(Debug)]
pub struct SnakeViewModel {
    model: Snake,
    bot_type: SnakeBotType,
}

impl SnakeViewModel {
    #[must_use]
    pub fn new(bot_type: SnakeBotType) -> Self {
        Self {
            model: Snake::new(),
            bot_type,
        }
    }

    // not the most scalable but number of types will remain small
    #[must_use]
    pub fn get_all_bot_types() -> [SnakeBotType; 1] {
        SnakeBotType::VALUES
    }

    pub fn make_bot_thread(&self) -> impl Stream<Item = Message> {
        let bot = make_new_bot(&self.bot_type);
        stream::channel(100, |mut output| async move {
            let (sender, mut receiver) = mpsc::channel::<ChannelMessage>(100);
            match output
                .send(Message::Snake(SnakeMessage::SnakeGameMessage(
                    SnakeGameMessage::ChannelMessage(ChannelMessage::BotReady(sender.clone())),
                )))
                .await
            {
                Ok(()) => (),
                Err(e) => {
                    debug!("Problem sending BotReady message: {}", e);
                }
            }
            loop {
                let input = receiver.next().await;
                if let Some(message) = input {
                    match message {
                        ChannelMessage::Input(grid) => {
                            match output
                                .send(Message::Snake(SnakeMessage::SnakeGameMessage(
                                    SnakeGameMessage::ChannelMessage(ChannelMessage::BotMove(
                                        bot.make_move(grid),
                                    )),
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
                            SnakeGameMessage::ChannelMessage(ChannelMessage::BotReady(
                                sender.clone(),
                            )),
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

    fn handle_channel_message(&mut self, message: ChannelMessage) {
        match message {
            ChannelMessage::BotReady(sender) => {
                if self.model.get_winner() != 0 {
                    let mut csend = sender.clone();
                    tokio::task::spawn(async move {
                        match csend.send(ChannelMessage::Kill).await {
                            Ok(()) => (),
                            Err(e) => debug!("Problem sending kill message: {e}"),
                        }
                    });
                }
                let bot_time_between_moves = make_new_bot(&self.bot_type).get_move_time();
                let millis_diff = Instant::now()
                    .duration_since(*self.model.get_second_player_last_move_time())
                    .as_millis();
                if millis_diff < u128::from(bot_time_between_moves)
                    || millis_diff < u128::from(snake_model::PLAYER_MOVEMENT_CAP)
                {
                    let max_wait = if snake_model::PLAYER_MOVEMENT_CAP > bot_time_between_moves {
                        u128::from(snake_model::PLAYER_MOVEMENT_CAP) - millis_diff
                    } else {
                        u128::from(bot_time_between_moves) - millis_diff
                    };
                    let mut csend = sender.clone();
                    tokio::spawn(async move {
                        match max_wait.try_into() {
                            Ok(w) => match csend.send(ChannelMessage::Idle(w)).await {
                                Ok(()) => (),
                                Err(e) => debug!("Problem sending idle message: {e}"),
                            },
                            Err(e) => {
                                warn!("Problem converting u128 to u64: {}.\nWaiting MILLIS_BETWEEN_FRAMES", e);
                                match csend
                                    .send(ChannelMessage::Idle(snake_model::MILLIS_BETWEEN_FRAMES))
                                    .await
                                {
                                    Ok(()) => (),
                                    Err(e) => debug!("Problem sending idle message: {e}"),
                                }
                            }
                        }
                    });
                } else {
                    let mut csend = sender.clone();
                    let grid = self.model.get_grid().clone();
                    tokio::spawn(async move {
                        match csend.send(ChannelMessage::Input(grid)).await {
                            Ok(()) => (),
                            Err(e) => debug!("Problem sending input message: {e}"),
                        }
                    });
                }
            }
            ChannelMessage::BotMove(bot_move) => {
                if self.model.get_winner() != 0 {
                    return;
                }
                self.model
                    .move_character(bot_move, 2, false, Instant::now());
            }
            _ => (),
        }
    }

    #[must_use]
    pub fn get_ref_backing_grid(&self) -> &Vec<Vec<SnakeBlock>> {
        self.model.get_grid()
    }

    #[must_use]
    pub fn get_winner(&self) -> u8 {
        self.model.get_winner()
    }

    #[must_use]
    pub fn get_time_between_frames(&self) -> u64 {
        snake_model::MILLIS_BETWEEN_FRAMES
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
                        if self.model.get_winner() != 0 {
                            if matches!(key, Key::Named(Named::Space)) {
                                return Some(Message::Snake(SnakeMessage::SnakeGameMessage(
                                    SnakeGameMessage::Reset,
                                )));
                            }
                            return None;
                        }
                        let now = Instant::now();
                        if now
                            .duration_since(*self.model.get_first_player_last_move_time())
                            .as_millis()
                            < u128::from(snake_model::PLAYER_MOVEMENT_CAP)
                        {
                            return None;
                        }
                        let movement = match key {
                            // here we have Vec<Vec<Block>> so (dx, dy) structure
                            // results in dx moving us vertically and dy moving us sideways
                            Key::Named(code) => match code {
                                Named::ArrowUp => Some((-1, 0)),
                                Named::ArrowDown => Some((1, 0)),
                                Named::ArrowLeft => Some((0, -1)),
                                Named::ArrowRight => Some((0, 1)),
                                _ => None,
                            },
                            Key::Character(ref c) => match c.as_str() {
                                "w" | "W" => Some((-1, 0)),
                                "s" | "S" => Some((1, 0)),
                                "a" | "A" => Some((0, -1)),
                                "d" | "D" => Some((0, 1)),
                                _ => None,
                            },
                            Key::Unidentified => None,
                        };

                        if let Some((dx, dy)) = movement {
                            self.model.move_character((dx, dy), 1, false, now);
                        }
                        None
                    }
                    SnakeGameMessage::Timer(_) => {
                        if self.model.get_winner() != 0 {
                            return None;
                        }
                        let first_to_get_ticked = rand::thread_rng().gen_range(1..=2);
                        let second_to_get_ticked = if first_to_get_ticked == 1 { 2 } else { 1 };
                        let now = Instant::now();
                        self.model
                            .move_character((0, 0), first_to_get_ticked, true, now);
                        if self.model.get_winner() == 0 {
                            self.model
                                .move_character((0, 0), second_to_get_ticked, true, now);
                        }
                        None
                    }
                    SnakeGameMessage::Reset => Some(Message::Snake(
                        SnakeMessage::SnakeGameScreenTransition(self.bot_type.clone()),
                    )),
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
