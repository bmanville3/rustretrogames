use iced::{Element, Subscription};

use crate::{
    view::View,
    views::{
        home::{Home, HomeMessage},
        snake::snake_mediator::{SnakeMediator, SnakeMessage},
    },
};

// https://docs.rs/iced/latest/i686-unknown-linux-gnu/iced/?search=command#scaling-applications
pub struct State {
    screen: Screen,
}

#[derive(Debug)]
enum Screen {
    Home(Home),
    Snake(SnakeMediator),
}

#[derive(Clone, Debug)]
pub enum Message {
    Home(HomeMessage),
    Snake(SnakeMessage),
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
        }
    }

    pub fn update(&mut self, message: Message) {
        if let Some(next) = self.screen.update(message) {
            match next {
                Message::Home(_) => {
                    self.screen = Screen::Home(Home::new());
                }
                Message::Snake(_) => {
                    self.screen = Screen::Snake(SnakeMediator::new());
                }
            }
        }
    }

    #[must_use]
    pub fn view(&self) -> Element<Message> {
        self.screen.view()
    }

    pub fn subscription(&self) -> Subscription<Message> {
        self.screen.subscription()
    }
}

impl Default for State {
    fn default() -> Self {
        Self::new()
    }
}
