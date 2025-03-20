use iced::{
    widget::{button, column, container, text},
    Alignment, Element, Length, Subscription,
};
use log::debug;

use crate::{app::Message, view::View};

use super::snake::snake_home::SnakeMessage;

#[derive(Clone, Debug)]
pub enum HomeMessage {
    Default,
    Snake,
}

impl HomeMessage {
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

#[derive(Debug)]
pub struct Home {}

impl Home {
    #[must_use]
    pub fn new() -> Self {
        Self {}
    }
}

impl View for Home {
    fn update(&mut self, message: Message) -> Option<Message> {
        if let Message::Home(message) = message {
            match message {
                HomeMessage::Snake => Some(Message::Snake(SnakeMessage::Default)),
                HomeMessage::Default => Some(Message::Home(HomeMessage::Default)),
            }
        } else {
            debug!("Received message for Home but was: {:#?}", message);
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
