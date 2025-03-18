use iced::{
    widget::{button, container, text},
    Element, Length,
};
use log::debug;

use crate::{app::Message, view::View};

#[derive(Clone, Debug)]
pub enum SudokuMessage {
    Default,
    Home,
}

impl SudokuMessage {
    #[must_use]
    pub fn new() -> Self {
        SudokuMessage::Default
    }
}

impl Default for SudokuMessage {
    fn default() -> Self {
        SudokuMessage::new()
    }
}

#[derive(Debug)]
pub struct Sudoku {}

impl Default for Sudoku {
    fn default() -> Self {
        Self::new()
    }
}

impl Sudoku {
    #[must_use]
    pub fn new() -> Self {
        Self {}
    }
}

impl View for Sudoku {
    fn update(&mut self, message: Message) -> Option<Message> {
        if let Message::Sudoku(message) = message {
            match message {
                SudokuMessage::Home => Some(Message::new_home()),
                // treating as refresh right now
                SudokuMessage::Default => Some(Message::new_pacman()),
            }
        } else {
            debug!("Received message for Sudoku but was: {:#?}", message);
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
            .on_press(Message::Sudoku(msg))
            .width(width)
            .height(height)
        };

        container(make_button("Go back to home", SudokuMessage::Home))
            .width(Length::Fill)
            .height(Length::Fill)
            .align_x(iced::alignment::Horizontal::Center)
            .align_y(iced::alignment::Vertical::Center)
            .into()
    }
}
