//! Module containing logic for the homescreen.

use iced::{
    widget::{button, column, container, text},
    Alignment, Element, Length, Subscription,
};
use log::{debug, warn};

use crate::{app::Message, view::View};

use super::snake::snake_mediator::SnakeMessage;

/// Message used for information propagation in [Home].
#[derive(Clone, Debug)]
pub enum HomeMessage {
    Default,
    Snake,
}

impl HomeMessage {
    /// Creates a new [`HomeMessage::Default`] message.
    #[must_use]
    pub fn new() -> Self {
        HomeMessage::Default
    }
}

impl Default for HomeMessage {
    fn default() -> Self {
        HomeMessage::new()
    }
}

/// Struct representing the homescreen.
#[derive(Debug)]
pub struct Home {}

impl Home {
    /// Creates a new [Home] struct.
    #[must_use]
    pub fn new() -> Self {
        Self {}
    }
}

impl View for Home {
    fn update(&mut self, message: Message) -> Option<Message> {
        debug!("New message at Home. Evaluating here.");
        if let Message::Home(message) = message {
            match message {
                HomeMessage::Snake => Some(Message::Snake(
                    SnakeMessage::SnakeSelectionScreenTransition((None, None)),
                )),
                HomeMessage::Default => Some(Message::Home(HomeMessage::Default)),
            }
        } else {
            warn!("Received message for Home but was should not have been sent here");
            None
        }
    }

    fn view(&self) -> Element<Message> {
        let height = 50;
        let width = height * 2;
        let make_button = |label, msg| {
            button(
                text(label)
                    .align_x(iced::alignment::Horizontal::Center)
                    .align_y(iced::alignment::Vertical::Center),
            )
            .on_press(Message::Home(msg))
            .width(width)
            .height(height)
        };

        let buttons = column![
            text("Welcome! Choose a game to start"),
            make_button("Snake", HomeMessage::Snake),
        ]
        .spacing(20)
        .align_x(Alignment::Center);

        container(buttons)
            .width(Length::Fill)
            .height(Length::Fill)
            .align_x(iced::alignment::Horizontal::Center)
            .align_y(iced::alignment::Vertical::Center)
            .into()
    }

    fn subscription(&self) -> iced::Subscription<Message> {
        Subscription::none()
    }
}

impl Default for Home {
    fn default() -> Self {
        Self::new()
    }
}
