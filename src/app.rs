//! Module to store state information.

use iced::{Element, Subscription};
use log::debug;

use crate::{
    view::View,
    views::{
        home::{Home, HomeMessage},
        snake::snake_mediator::{SnakeMediator, SnakeMessage},
    },
};

// https://docs.rs/iced/latest/i686-unknown-linux-gnu/iced/?search=command#scaling-applications
/// A struct holding the current [Screen] of the application.
pub struct State {
    screen: Screen,
}

/// Enum to encapsulate the [State]'s screen.
#[derive(Debug)]
enum Screen {
    Home(Home),
    Snake(SnakeMediator),
}

/// Enum of screen messages used to propagate information.
#[derive(Clone, Debug)]
pub enum Message {
    Home(HomeMessage),
    Snake(SnakeMessage),
}

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
    /// Creates a new [State] starting at the home [Screen].
    #[must_use]
    pub fn new() -> Self {
        Self {
            screen: Screen::Home(Home::new()),
        }
    }

    /// Updates the state's [Screen].
    /// Switches screens if state.screen.update returns a [Message].
    pub fn update(&mut self, message: Message) {
        debug!("New message at App level. Sending down...");
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

    /// Displays the state's [Screen].
    #[must_use]
    pub fn view(&self) -> Element<Message> {
        self.screen.view()
    }

    /// Subscribes to the state [Screen]'s [Subscription].
    pub fn subscription(&self) -> Subscription<Message> {
        self.screen.subscription()
    }
}

impl Default for State {
    fn default() -> Self {
        Self::new()
    }
}
