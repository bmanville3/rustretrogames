use iced::time::Instant;
use rand::{seq::SliceRandom, Rng};
use std::collections::VecDeque;
use std::marker::PhantomData;
use std::thread;
use std::time::Duration;
use tokio;

use iced::{
    futures::{channel::mpsc, SinkExt, Stream, StreamExt},
    keyboard::{key::Named, Key},
    stream,
    widget::{button, column, container, row, text, Column, Row},
    Border, Color, Element, Length, Subscription,
};
use log::{debug, warn};

use crate::{app::Message, bots::snake::snake_bot::SnakeBot, view::View, views::home::HomeMessage};

/// Amount of time before snake is forced to move.
pub const MILLIS_BETWEEN_FRAMES: u64 = 300;
/// Cap of how fast a player can make consecutive moves.
pub const PLAYER_MOVEMENT_CAP: u64 = MILLIS_BETWEEN_FRAMES / 10;

#[derive(Clone, Debug)]
pub enum SnakeMessage {
    Default,
    Home,
    ChannelMessage(ChannelMessage),
}

impl SnakeMessage {
    #[must_use]
    pub fn new() -> Self {
        SnakeMessage::Default
    }
}

impl Default for SnakeMessage {
    fn default() -> Self {
        SnakeMessage::new()
    }
}

#[derive(Clone, Debug)]
pub enum SnakeBlock {
    EMPTY,
    APPLE,
    PLAYERONE,
    HEADONE,
    PLAYERTWO,
    HEADTWO,
}

#[derive(Clone, Debug)]
pub enum ChannelMessage {
    BotReady(mpsc::Sender<ChannelMessage>),
    Input(Vec<Vec<SnakeBlock>>),
    Idle(u64),
    BotMove((i8, i8)),
    Kill,
}

#[derive(Debug)]
pub struct Snake<T>
where
    T: SnakeBot + std::marker::Send + 'static,
{
    grid: Vec<Vec<SnakeBlock>>,
    snake_one: VecDeque<(usize, usize)>,
    so_dir: (i8, i8),
    so_last_move: Instant,
    snake_two: VecDeque<(usize, usize)>,
    st_dir: (i8, i8),
    st_last_move: Instant,
    size: u8,
    winner: u8,
    bot_time_between_moves: u64,
    pub sub_key: u64,
    phantom: PhantomData<T>,
}

impl<T> Snake<T>
where
    T: SnakeBot + std::marker::Send + 'static,
{
    /// # Panics
    ///
    /// Will panic if size does not fit in '[u8]'.
    #[must_use]
    pub fn new() -> Self {
        let size = 32;
        let mut grid: Vec<Vec<SnakeBlock>> = Vec::with_capacity(size);
        for _ in 0..size {
            let mut row: Vec<SnakeBlock> = Vec::with_capacity(size);
            for _ in 0..size {
                row.push(SnakeBlock::EMPTY);
            }
            grid.push(row);
        }
        let mut snake_one = VecDeque::new();
        let player_one_loc = size / 4;
        grid[player_one_loc + 3][player_one_loc + 3] = SnakeBlock::APPLE;
        grid[player_one_loc][player_one_loc] = SnakeBlock::HEADONE;

        let mut snake_two = VecDeque::new();
        let player_two_loc = size * 3 / 4;
        grid[player_two_loc - 3][player_two_loc - 3] = SnakeBlock::APPLE;
        grid[player_two_loc][player_two_loc] = SnakeBlock::HEADTWO;

        snake_one.push_front((player_one_loc, player_one_loc));
        snake_two.push_front((player_two_loc, player_two_loc));

        Self {
            grid,
            snake_one,
            so_dir: (1, 0),
            so_last_move: Instant::now(),
            snake_two,
            st_dir: (-1, 0),
            st_last_move: Instant::now(),
            size: size.try_into().unwrap(),
            winner: 0,
            bot_time_between_moves: Self::make_bot().get_move_time(),
            sub_key: 0,
            phantom: PhantomData,
        }
    }

    fn make_bot() -> T {
        T::new()
    }

    fn make_bot_thread() -> impl Stream<Item = Message> {
        stream::channel(100, |mut output| async move {
            let (sender, mut receiver) = mpsc::channel::<ChannelMessage>(100);
            let bot = Self::make_bot();
            match output
                .send(Message::Snake(SnakeMessage::ChannelMessage(
                    ChannelMessage::BotReady(sender.clone()),
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
                                .send(Message::Snake(SnakeMessage::ChannelMessage(
                                    ChannelMessage::BotMove(bot.make_move(grid)),
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
                        .send(Message::Snake(SnakeMessage::ChannelMessage(
                            ChannelMessage::BotReady(sender.clone()),
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

    fn put_random_apple(&mut self) {
        let mut avaliable = Vec::new();
        for i in 0..self.grid.len() {
            for j in 0..self.grid[i].len() {
                if matches!(self.grid[i][j], SnakeBlock::EMPTY) {
                    avaliable.push((i, j));
                }
            }
        }
        if avaliable.is_empty() {
            warn!("No where left to place apples");
            return;
        }
        let index = avaliable.choose(&mut rand::thread_rng());
        if let Some(index) = index {
            self.grid[index.0][index.1] = SnakeBlock::APPLE;
        } else {
            warn!("Unable to get a random index. Choosing middle one found");
            let index = avaliable[avaliable.len() / 2 + 1];
            self.grid[index.0][index.1] = SnakeBlock::APPLE;
        }
    }

    fn move_character(
        &mut self,
        delta: (i8, i8),
        character: u8,
        by_timer: bool,
        time: Instant,
    ) -> u8 {
        let mut delta = delta;
        let opposition;
        let snake;
        let head_type;
        let block_type;
        if character == 1 {
            // cant do an insta turn around
            if delta == self.so_dir || (-delta.0, -delta.1) == self.so_dir {
                return 0;
            }
            opposition = 2;
            snake = &mut self.snake_one;
            block_type = SnakeBlock::PLAYERONE;
            head_type = SnakeBlock::HEADONE;
            if by_timer {
                delta = self.so_dir;
            } else {
                self.so_dir = delta;
                self.so_last_move = time;
            }
        } else {
            // cant do an insta turn around
            if delta == self.st_dir || (-delta.0, -delta.1) == self.st_dir {
                return 0;
            }
            opposition = 1;
            snake = &mut self.snake_two;
            block_type = SnakeBlock::PLAYERTWO;
            head_type = SnakeBlock::HEADTWO;
            if by_timer {
                delta = self.st_dir;
            } else {
                self.st_dir = delta;
                self.st_last_move = time;
            }
        }

        let front = snake.front().unwrap();
        // the grid and snake is bound by u8 so i64 should always cover what is needed
        let new_x: i64 = i64::try_from(front.0).unwrap() + i64::from(delta.0);
        let new_y: i64 = i64::try_from(front.1).unwrap() + i64::from(delta.1);
        let size = i64::from(self.size);
        let mut winner = 0;
        if new_x < size && new_y < size && new_x >= 0 && new_y >= 0 {
            let new_x = usize::try_from(new_x).unwrap();
            let new_y = usize::try_from(new_y).unwrap();
            match self.grid.get(new_x).unwrap().get(new_y).unwrap() {
                SnakeBlock::APPLE => {
                    self.grid[new_x][new_y] = head_type;
                    self.grid[front.0][front.1] = block_type;
                    snake.push_front((new_x, new_y));
                    self.put_random_apple();
                }
                SnakeBlock::EMPTY => {
                    self.grid[new_x][new_y] = head_type;
                    self.grid[front.0][front.1] = block_type;
                    snake.push_front((new_x, new_y));
                    if let Some(old_tail) = snake.pop_back() {
                        self.grid[old_tail.0][old_tail.1] = SnakeBlock::EMPTY;
                    } else {
                        debug!("Removed from back but got none");
                    }
                }
                _ => {
                    winner = opposition;
                }
            }
        } else {
            winner = opposition;
        }
        winner
    }

    fn handle_channel_message(&mut self, message: ChannelMessage) {
        match message {
            ChannelMessage::BotReady(sender) => {
                if self.winner != 0 {
                    let mut csend = sender.clone();
                    tokio::task::spawn(async move {
                        match csend.send(ChannelMessage::Kill).await {
                            Ok(()) => (),
                            Err(e) => debug!("Problem sending kill message: {e}"),
                        }
                    });
                }
                let millis_diff = Instant::now().duration_since(self.st_last_move).as_millis();
                if millis_diff < u128::from(self.bot_time_between_moves)
                    || millis_diff < u128::from(PLAYER_MOVEMENT_CAP)
                {
                    let max_wait = if PLAYER_MOVEMENT_CAP > self.bot_time_between_moves {
                        u128::from(PLAYER_MOVEMENT_CAP) - millis_diff
                    } else {
                        u128::from(self.bot_time_between_moves) - millis_diff
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
                                    .send(ChannelMessage::Idle(MILLIS_BETWEEN_FRAMES))
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
                    let grid = self.grid.clone();
                    tokio::spawn(async move {
                        match csend.send(ChannelMessage::Input(grid)).await {
                            Ok(()) => (),
                            Err(e) => debug!("Problem sending input message: {e}"),
                        }
                    });
                }
            }
            ChannelMessage::BotMove(bot_move) => {
                if self.winner != 0 {
                    return;
                }
                self.winner = self.move_character(bot_move, 2, false, Instant::now());
            }
            _ => (),
        }
    }
}

impl<T: SnakeBot + std::marker::Send + 'static> Default for Snake<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: SnakeBot + std::marker::Send + 'static> View for Snake<T> {
    fn update(&mut self, message: Message) -> Option<Message> {
        if let Message::Snake(message) = message {
            match message {
                SnakeMessage::Home => Some(Message::Home(HomeMessage::Default)),
                // treating as refresh right now
                SnakeMessage::Default => Some(Message::Snake(SnakeMessage::Default)),
                SnakeMessage::ChannelMessage(message) => {
                    self.handle_channel_message(message);
                    None
                }
            }
        } else if let Message::KeyPressed(key) = message {
            if self.winner != 0 {
                if matches!(key, Key::Named(Named::Space)) {
                    return Some(Message::Snake(SnakeMessage::Default));
                }
                return None;
            }
            let now = Instant::now();
            if now.duration_since(self.so_last_move).as_millis() < u128::from(PLAYER_MOVEMENT_CAP) {
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
                self.winner = self.move_character((dx, dy), 1, false, now);
            }
            None
        } else if let Message::Timer(_) = message {
            if self.winner != 0 {
                return None;
            }
            let first_to_get_ticked = rand::thread_rng().gen_range(1..=2);
            let second_to_get_ticked = if first_to_get_ticked == 1 { 2 } else { 1 };
            let now = Instant::now();
            self.winner = self.move_character((0, 0), first_to_get_ticked, true, now);
            if self.winner == 0 {
                self.winner = self.move_character((0, 0), second_to_get_ticked, true, now);
            }
            None
        } else {
            debug!("Received message for Snake but was: {:#?}", message);
            None
        }
    }

    fn view(&self) -> Element<Message> {
        let mut grid_view = Column::new();
        let cell_size = 20;

        let make_container = |color: Color| {
            container(text(" ").color(color)) // Empty text to preserve size
                .width(cell_size)
                .height(cell_size)
                .style(move |_: &_| container::Style {
                    // TODO Fix this
                    border: Border {
                        color: Color::from_rgba(0.0, 0.0, 0.0, 0.1),
                        width: 1.0,
                        ..Default::default()
                    },
                    background: Some(color.into()),
                    ..container::Style::default()
                })
        };

        for grid_row in &self.grid {
            let mut row = Row::new();
            for entry in grid_row {
                let rectangle = match entry {
                    SnakeBlock::EMPTY => make_container(Color::WHITE),
                    SnakeBlock::APPLE => make_container(Color::from_rgb(1.0, 0.0, 0.0)),
                    SnakeBlock::PLAYERONE => make_container(Color::from_rgba(0.0, 1.0, 0.0, 0.8)),
                    SnakeBlock::PLAYERTWO => make_container(Color::from_rgba(0.0, 0.0, 1.0, 0.8)),
                    SnakeBlock::HEADONE => make_container(Color::from_rgb(0.0, 1.0, 0.0)),
                    SnakeBlock::HEADTWO => make_container(Color::from_rgb(0.0, 0.0, 1.0)),
                };

                row = row.push(rectangle);
            }
            grid_view = grid_view.push(row);
        }

        // "Go back to home" button
        let home_button = button(text("Back to Home"))
            .on_press(Message::Snake(SnakeMessage::Home))
            .width(160)
            .height(40);
        let restart_button = button(text("Restart"))
            .on_press(Message::Snake(SnakeMessage::Default))
            .width(80)
            .height(40);

        let game = container(
            column![
                row![home_button, restart_button].spacing(10), // Keep the home button at the top left
                grid_view,                                     // Below it, display the game grid
            ]
            .spacing(10),
        )
        .width(Length::Fill)
        .height(Length::Fill)
        .align_x(iced::alignment::Horizontal::Center)
        .align_y(iced::alignment::Vertical::Center);
        if self.winner != 0 {
            return column!(
                game,
                text(format!(
                    "GAME OVER. YOU {}!",
                    if self.winner == 2 { "LOST" } else { "WON" }
                ))
            )
            .align_x(iced::alignment::Horizontal::Center)
            .into();
        }
        game.into()
    }

    fn subscription(&self) -> Subscription<Message> {
        Subscription::run_with_id(self.sub_key, Self::make_bot_thread())
    }
}
