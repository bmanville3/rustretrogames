use iced::{
    widget::{button, column, container, row, text},
    Alignment, Element, Length,
};
use log::debug;

use crate::{app::Message, view::View};

#[derive(Clone, Debug)]
pub enum HomeMessage {
    Default,
    PacMan,
    Pong,
    Snake,
    Sudoku,
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
                HomeMessage::PacMan => Some(Message::new_pacman()),
                HomeMessage::Pong => Some(Message::new_pong()),
                HomeMessage::Snake => Some(Message::new_snake()),
                HomeMessage::Sudoku => Some(Message::new_sudoku()),
                // we will treat this as refreshing
                HomeMessage::Default => Some(Message::new_home()),
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
            row![
                make_button("Pac-Man", HomeMessage::PacMan),
                make_button("Pong", HomeMessage::Pong),
            ]
            .spacing(10),
            row![
                make_button("Snake", HomeMessage::Snake),
                make_button("Sudoku", HomeMessage::Sudoku),
            ]
            .spacing(10),
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
}

impl Default for Home {
    fn default() -> Self {
        Self::new()
    }
}
