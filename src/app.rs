use iced::keyboard::{self, Key};
use iced::time::{self, Duration, Instant};
use iced::{Element, Subscription};
use log::debug;

use crate::bots::snake::random_bot::RandomBot;
use crate::{
    view::View,
    views::{
        home::{Home, HomeMessage},
        snake::snake_home::{self, Snake, SnakeMessage},
    },
};

// https://docs.rs/iced/latest/i686-unknown-linux-gnu/iced/?search=command#scaling-applications
pub struct State {
    screen: Screen,
    millis_between_frames: u64,
    counter: u64,
}

#[derive(Debug)]
enum Screen {
    Home(Home),
    Snake(Snake<RandomBot>),
}

#[derive(Clone, Debug)]
pub enum Message {
    Home(HomeMessage),
    Snake(SnakeMessage),
    KeyPressed(Key),
    Timer(Instant),
}

// Implement `View` for `Screen`
impl View for Screen {
    fn update(&mut self, message: Message) -> Option<Message> {
        match (self, message) {
            (Screen::Home(screen), msg) => screen.update(msg),
            (Screen::Snake(screen), msg) => screen.update(msg),
        }
    }

    fn view(&self) -> Element<Message> {
        match self {
            Screen::Home(screen) => screen.view(),
            Screen::Snake(screen) => screen.view(),
        }
    }

    fn subscription(&self) -> Subscription<Message> {
        match self {
            Screen::Home(screen) => screen.subscription(),
            Screen::Snake(screen) => screen.subscription(),
        }
    }
}

impl State {
    #[must_use]
    pub fn new() -> Self {
        Self {
            screen: Screen::Home(Home::new()),
            millis_between_frames: 0,
            counter: 0,
        }
    }

    pub fn update(&mut self, message: Message) {
        #[cfg(debug_assertions)]
        {
            if let Message::Timer(_) = &message {
                // false if statement
            } else {
                debug!("Update message in State: {:#?}", message);
            }
        }
        if let Some(next) = self.screen.update(message) {
            match next {
                Message::Home(_) => {
                    self.screen = {
                        self.millis_between_frames = 0;
                        Screen::Home(Home::new())
                    }
                }
                Message::Snake(_) => {
                    self.screen = {
                        self.millis_between_frames = snake_home::MILLIS_BETWEEN_FRAMES;
                        let mut game = Snake::new();
                        // if overflow ever happens (which it shouldnt), it should be fine as the subscriptions will already be dropped
                        // there are 2^64 - 1 ids avaliable so...
                        self.counter += 1;
                        game.sub_key = self.counter;
                        Screen::Snake(game)
                    }
                }
                _ => (),
            }
        }
    }

    #[must_use]
    pub fn view(&self) -> Element<Message> {
        self.screen.view()
    }

    pub fn subscription(&self) -> Subscription<Message> {
        let tick = if self.millis_between_frames == 0 {
            Subscription::none()
        } else {
            time::every(Duration::from_millis(self.millis_between_frames)).map(Message::Timer)
        };

        let keyboard = keyboard::on_key_press(|key, _| Some(Message::KeyPressed(key)));

        let additional_subscription = self.screen.subscription();

        Subscription::batch(vec![tick, keyboard, additional_subscription])
    }
}

impl Default for State {
    fn default() -> Self {
        Self::new()
    }
}
