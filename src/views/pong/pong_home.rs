use iced::{
    widget::{button, container, text},
    Element, Length,
};
use log::debug;

use crate::{app::Message, view::View};

#[derive(Clone, Debug)]
pub enum PongMessage {
    Default,
    Home,
}

impl PongMessage {
    #[must_use]
    pub fn new() -> Self {
        PongMessage::Default
    }
}

impl Default for PongMessage {
    fn default() -> Self {
        PongMessage::new()
    }
}

#[derive(Debug)]
pub struct Pong {}

impl Default for Pong {
    fn default() -> Self {
        Self::new()
    }
}

impl Pong {
    #[must_use]
    pub fn new() -> Self {
        Self {}
    }
}

impl View for Pong {
    fn update(&mut self, message: Message) -> Option<Message> {
        if let Message::Pong(message) = message {
            match message {
                PongMessage::Home => Some(Message::new_home()),
                // treating as refresh right now
                PongMessage::Default => Some(Message::new_pacman()),
            }
        } else {
            debug!("Received message for Pong but was: {:#?}", message);
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
            .on_press(Message::Pong(msg))
            .width(width)
            .height(height)
        };

        container(make_button("Go back to home", PongMessage::Home))
            .width(Length::Fill)
            .height(Length::Fill)
            .align_x(iced::alignment::Horizontal::Center)
            .align_y(iced::alignment::Vertical::Center)
            .into()
    }
}
