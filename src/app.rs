use iced::Element;

use crate::{
    view::View,
    views::{
        home::{Home, HomeMessage},
        pac_man::pac_man_home::{PacMan, PacManMessage},
        pong::pong_home::{Pong, PongMessage},
        snake::snake_home::{Snake, SnakeMessage},
        sudoku::sudoku_home::{Sudoku, SudokuMessage},
    },
};

// https://docs.rs/iced/latest/i686-unknown-linux-gnu/iced/?search=command#scaling-applications
pub struct State {
    screen: Screen,
}

#[derive(Debug)]
enum Screen {
    Home(Home),
    PacMan(PacMan),
    Pong(Pong),
    Snake(Snake),
    Sudoku(Sudoku),
}

impl Screen {
    pub fn new_home() -> Self {
        Screen::Home(Home::new())
    }

    pub fn new_pacman() -> Self {
        Screen::PacMan(PacMan::new())
    }

    pub fn new_pong() -> Self {
        Screen::Pong(Pong::new())
    }

    pub fn new_snake() -> Self {
        Screen::Snake(Snake::new())
    }

    pub fn new_sudoku() -> Self {
        Screen::Sudoku(Sudoku::new())
    }
}

#[derive(Clone, Debug)]
pub enum Message {
    Home(HomeMessage),
    PacMan(PacManMessage),
    Pong(PongMessage),
    Snake(SnakeMessage),
    Sudoku(SudokuMessage),
}

impl Message {
    #[must_use]
    pub fn new_home() -> Self {
        Message::Home(HomeMessage::new())
    }

    #[must_use]
    pub fn new_pacman() -> Self {
        Message::PacMan(PacManMessage::new())
    }

    #[must_use]
    pub fn new_pong() -> Self {
        Message::Pong(PongMessage::new())
    }

    #[must_use]
    pub fn new_snake() -> Self {
        Message::Snake(SnakeMessage::new())
    }

    #[must_use]
    pub fn new_sudoku() -> Self {
        Message::Sudoku(SudokuMessage::new())
    }
}

// Implement `View` for `Screen`
impl View for Screen {
    fn update(&mut self, message: Message) -> Option<Message> {
        match (self, message) {
            (Screen::Home(screen), msg) => screen.update(msg),
            (Screen::PacMan(screen), msg) => screen.update(msg),
            (Screen::Pong(screen), msg) => screen.update(msg),
            (Screen::Snake(screen), msg) => screen.update(msg),
            (Screen::Sudoku(screen), msg) => screen.update(msg),
        }
    }

    fn view(&self) -> Element<Message> {
        match self {
            Screen::Home(screen) => screen.view(),
            Screen::PacMan(screen) => screen.view(),
            Screen::Pong(screen) => screen.view(),
            Screen::Snake(screen) => screen.view(),
            Screen::Sudoku(screen) => screen.view(),
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

    pub fn update(state: &mut State, message: Message) {
        if let Some(next) = state.screen.update(message) {
            match next {
                Message::Home(_) => state.screen = Screen::new_home(),
                Message::PacMan(_) => state.screen = Screen::new_pacman(),
                Message::Pong(_) => state.screen = Screen::new_pong(),
                Message::Snake(_) => state.screen = Screen::new_snake(),
                Message::Sudoku(_) => state.screen = Screen::new_sudoku(),
            }
        }
    }

    #[must_use]
    pub fn view(state: &State) -> Element<Message> {
        state.screen.view()
    }
}

impl Default for State {
    fn default() -> Self {
        Self::new()
    }
}
