//! Module to handle switching between [`SnakeScreen`]s.

use iced::{Element, Subscription};
use log::{debug, error, warn};

use crate::{
    app::Message,
    view::View,
    view_models::snake::snake_view_model::{SnakeParams, SnakeViewModel},
    views::home::HomeMessage,
};

use super::{
    snake_game_screen::{SnakeGameMessage, SnakeGameScreen},
    snake_selection_screen::{SnakeSelectionMessage, SnakeSelectionScreen},
};

#[derive(Clone, Debug)]
pub enum SnakeMessage {
    SnakeGameScreenTransition(SnakeParams),
    SnakeSelectionScreenTransition((Option<SnakeParams>, Option<String>)),
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
    key: usize,
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
        debug!("Received message in SnakeMediator. Evaluating here.");
        if let Message::Snake(snake_message) = message {
            match snake_message {
                SnakeMessage::SnakeGameScreenTransition(params) => {
                    debug!("Transitioning to Snake Game Screen...");
                    let new_vm = match SnakeViewModel::new(params.clone()) {
                        Ok(vm) => vm,
                        Err(e) => {
                            error!("Error when creating Snake View Model: {:#?}", e);
                            return Some(Message::Snake(
                                SnakeMessage::SnakeSelectionScreenTransition((
                                    Some(params),
                                    Some(format!("Error when creating Snake View Model: {e:#?}")),
                                )),
                            ));
                        }
                    };
                    self.key += 1;
                    self.snake_screen =
                        SnakeScreen::SnakeGameScreen(SnakeGameScreen::new(new_vm, self.key));
                    None
                }
                SnakeMessage::SnakeSelectionScreenTransition((params, message)) => {
                    debug!("Transitioning to Snake Selection Screen...");
                    let mut sss = if let Some(p) = params {
                        SnakeSelectionScreen::from_params(p)
                    } else {
                        SnakeSelectionScreen::new()
                    };
                    sss.add_optional_message(message);
                    self.snake_screen = SnakeScreen::SnakeSelectionScreen(sss);
                    None
                }
                SnakeMessage::HomeScreenTransition => {
                    debug!("Transitioning to Home scnreen...");
                    Some(Message::Home(HomeMessage::Default))
                }
                _ => {
                    debug!("Received message at SnakeMediator. Sending down...");
                    // the basic idea of the following code is we do the normal update with the screen
                    // if we need to transition away from the screen though, we call update on ourselves
                    // with the transition as this module is used to transition between screens
                    // we should really only ever hit one level of recursion as every case is caught above for transitioning
                    let mut snake_message = self.snake_screen.update(Message::Snake(snake_message));
                    snake_message.as_ref()?;
                    let mut i = 0;
                    while let Some(new_message) = self.update(snake_message.unwrap()) {
                        i += 1;
                        if i > 20 {
                            error!(
                                "Did 20 recurssive loops. Giving up. Just going back to HomeScreen"
                            );
                            return Some(Message::Home(HomeMessage::Default));
                        }
                        snake_message = Some(new_message);
                    }
                    None
                }
            }
        } else {
            warn!("Received a non-snake message in the snake mediator.");
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
