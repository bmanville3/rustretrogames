use iced::{Element, Subscription};
use log::debug;

use crate::{
    app::Message, models::snake::snake_bot::SnakeBotType, view::View,
    view_models::snake::snake_view_model::SnakeViewModel, views::home::HomeMessage,
};

use super::{
    snake_game_screen::{SnakeGameMessage, SnakeGameScreen},
    snake_selection_screen::{SnakeSelectionMessage, SnakeSelectionScreen},
};

#[derive(Clone, Debug)]
pub enum SnakeMessage {
    SnakeGameScreenTransition(SnakeBotType),
    SnakeSelectionScreenTransition,
    HomeScreenTransition,
    SnakeGameMessage(SnakeGameMessage),
    SnakeSelectionMessage(SnakeSelectionMessage),
}

#[derive(Debug)]
pub enum SnakeScreen {
    SnakeGameScreen(SnakeGameScreen),
    SnakeSelectionScreen(SnakeSelectionScreen),
}

impl View for SnakeScreen {
    fn update(&mut self, message: Message) -> Option<Message> {
        match self {
            SnakeScreen::SnakeGameScreen(snake_game_screen) => snake_game_screen.update(message),
            SnakeScreen::SnakeSelectionScreen(snake_selection_screen) => {
                snake_selection_screen.update(message)
            }
        }
    }

    fn view(&self) -> Element<Message> {
        match self {
            SnakeScreen::SnakeGameScreen(snake_game_screen) => snake_game_screen.view(),
            SnakeScreen::SnakeSelectionScreen(snake_selection_screen) => {
                snake_selection_screen.view()
            }
        }
    }

    fn subscription(&self) -> Subscription<Message> {
        match self {
            SnakeScreen::SnakeGameScreen(snake_game_screen) => snake_game_screen.subscription(),
            SnakeScreen::SnakeSelectionScreen(snake_selection_screen) => {
                snake_selection_screen.subscription()
            }
        }
    }
}

#[derive(Debug)]
pub struct SnakeMediator {
    snake_screen: SnakeScreen,
    key: u64,
}

impl Default for SnakeMediator {
    fn default() -> Self {
        Self::new()
    }
}

impl SnakeMediator {
    #[must_use]
    pub fn new() -> Self {
        Self {
            snake_screen: SnakeScreen::SnakeSelectionScreen(SnakeSelectionScreen::new()),
            key: 0,
        }
    }
}

impl View for SnakeMediator {
    fn update(&mut self, message: Message) -> Option<Message> {
        if let Message::Snake(snake_message) = message {
            match snake_message {
                SnakeMessage::SnakeGameScreenTransition(bot_type) => {
                    debug!("Transitioning to snake game screen");
                    self.key += 1;
                    self.snake_screen = SnakeScreen::SnakeGameScreen(SnakeGameScreen::new(
                        SnakeViewModel::new(bot_type),
                        self.key,
                    ));
                    None
                }
                SnakeMessage::SnakeSelectionScreenTransition => {
                    debug!("Transitioning to snake selection screen");
                    self.snake_screen =
                        SnakeScreen::SnakeSelectionScreen(SnakeSelectionScreen::new());
                    None
                }
                SnakeMessage::HomeScreenTransition => {
                    debug!("Transitioning to home scnreen");
                    Some(Message::Home(HomeMessage::Default))
                }
                _ => match self.snake_screen.update(Message::Snake(snake_message)) {
                    // will be finite recursion (should only be depth 1 for transitioning to new screen)
                    // the screens only return SnakeMessages so they will be hit in the cases above this one
                    Some(m) => self.update(m),
                    None => None,
                },
            }
        } else {
            debug!(
                "Received a non-snake message in the snake mediator. Message: {:#?}",
                message
            );
            None
        }
    }

    fn view(&self) -> Element<Message> {
        self.snake_screen.view()
    }

    fn subscription(&self) -> Subscription<Message> {
        self.snake_screen.subscription()
    }
}
