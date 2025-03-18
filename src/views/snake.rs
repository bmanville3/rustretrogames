use iced::{
    widget::{button, container, text},
    Element, Length,
};
use log::debug;

use crate::{app::Message, view::View};

#[derive(Clone, Debug)]
pub enum SnakeMessage {
    Default,
    Home,
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

#[derive(Debug)]
pub struct Snake {}

impl Default for Snake {
    fn default() -> Self {
        Self::new()
    }
}

impl Snake {
    #[must_use]
    pub fn new() -> Self {
        Self {}
    }
}

impl View for Snake {
    fn update(&mut self, message: Message) -> Option<Message> {
        if let Message::Snake(message) = message {
            match message {
                SnakeMessage::Home => Some(Message::new_home()),
                // treating as refresh right now
                SnakeMessage::Default => Some(Message::new_pacman()),
            }
        } else {
            debug!("Received message for Snake but was: {:#?}", message);
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
            .on_press(Message::Snake(msg))
            .width(width)
            .height(height)
        };

        container(make_button("Go back to home", SnakeMessage::Home))
            .width(Length::Fill)
            .height(Length::Fill)
            .align_x(iced::alignment::Horizontal::Center)
            .align_y(iced::alignment::Vertical::Center)
            .into()
    }
}
