use iced::{
    widget::{button, container, text},
    Element, Length,
};
use log::debug;

use crate::{app::Message, view::View};

#[derive(Clone, Debug)]
pub enum PacManMessage {
    Default,
    Home,
}

impl PacManMessage {
    #[must_use]
    pub fn new() -> Self {
        PacManMessage::Default
    }
}

impl Default for PacManMessage {
    fn default() -> Self {
        PacManMessage::new()
    }
}

#[derive(Debug)]
pub struct PacMan {}

impl Default for PacMan {
    fn default() -> Self {
        Self::new()
    }
}

impl PacMan {
    #[must_use]
    pub fn new() -> Self {
        Self {}
    }
}

impl View for PacMan {
    fn update(&mut self, message: Message) -> Option<Message> {
        if let Message::PacMan(message) = message {
            match message {
                PacManMessage::Home => Some(Message::new_home()),
                // treating as refresh right now
                PacManMessage::Default => Some(Message::new_pacman()),
            }
        } else {
            debug!("Received message for PacMan but was: {:#?}", message);
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
            .on_press(Message::PacMan(msg))
            .width(width)
            .height(height)
        };

        container(make_button("Go back to home", PacManMessage::Home))
            .width(Length::Fill)
            .height(Length::Fill)
            .align_x(iced::alignment::Horizontal::Center)
            .align_y(iced::alignment::Vertical::Center)
            .into()
    }
}
