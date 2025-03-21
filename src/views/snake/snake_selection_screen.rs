//! Module to handle Snake selection logic.

use iced::{
    alignment,
    widget::{button, column, container, pick_list, text},
    Element, Length, Subscription,
};
use log::{debug, info};

use crate::{
    app::Message,
    models::snake::snake_bot::SnakeBotType,
    view::View,
    view_models::snake::{
        snake_selection_view_model::SnakeSelectionViewModel, snake_view_model::SnakeParams,
    },
};

use super::snake_mediator::SnakeMessage;

/// Message to propagate information in the [SnakeSelectionScreen].
#[derive(Debug, Clone)]
pub enum SnakeSelectionMessage {
    BotTypeSelected(SnakeBotType),
    NumberOfBots(usize),
    NumberOfReal(usize),
    GridSize(usize),
    Submit,
}

/// Struct representing the Snake selection screen.
#[derive(Debug)]
pub struct SnakeSelectionScreen {
    view_model: SnakeSelectionViewModel,
    /// [SnakeBotType] to play Snake against.
    selected_bot: Option<SnakeBotType>,
    number_of_real_players: Option<usize>,
    number_of_bots: Option<usize>,
    grid_size: Option<usize>,
    additional_message: Option<String>,
}

impl Default for SnakeSelectionScreen {
    fn default() -> Self {
        Self::new()
    }
}


// TODO: Make it so you dont have to select a bot (single player/multiplayer with friends only)



impl SnakeSelectionScreen {
    /// Creates a new struct with no selected [SnakeBotType].
    #[must_use]
    pub fn new() -> Self {
        Self {
            view_model: SnakeSelectionViewModel {},
            selected_bot: None,
            number_of_real_players: None,
            number_of_bots: None,
            grid_size: None,
            additional_message: None,
        }
    }

    #[must_use]
    pub fn from_params(params: SnakeParams) -> Self {
        let mut sss = SnakeSelectionScreen::new();
        // act like sending messages to update with existing stuff
        sss.update(Message::Snake(SnakeMessage::SnakeSelectionMessage(SnakeSelectionMessage::BotTypeSelected(params.bot_type))));
        sss.update(Message::Snake(SnakeMessage::SnakeSelectionMessage(SnakeSelectionMessage::NumberOfBots(params.number_of_bots))));
        sss.update(Message::Snake(SnakeMessage::SnakeSelectionMessage(SnakeSelectionMessage::NumberOfReal(params.number_of_real_players))));
        sss.update(Message::Snake(SnakeMessage::SnakeSelectionMessage(SnakeSelectionMessage::GridSize(params.grid_size))));
        sss.additional_message = None;
        sss
    }

    pub fn add_message(&mut self, message: String) {
        self.additional_message = Some(message);
    }

    pub fn add_optional_message(&mut self, message: Option<String>) {
        self.additional_message = message;
    }
}

impl View for SnakeSelectionScreen {
    fn update(&mut self, message: Message) -> Option<Message> {
        if let Message::Snake(snake_message) = message {
            if let SnakeMessage::SnakeSelectionMessage(ssm) = snake_message {
                self.additional_message = None;
                match ssm {
                    SnakeSelectionMessage::BotTypeSelected(bot_type) => {
                        info!("Selected bot: {}", bot_type);
                        self.selected_bot = Some(bot_type);
                    }
                    SnakeSelectionMessage::NumberOfBots(nob) => {
                        if !self.view_model.validate_number_of_bots(nob) {
                            self.additional_message =
                                Some(format!("Selected an invalid number of bots: {nob}"));
                            return None;
                        }
                        info!("Selected number of bots: {}", nob);
                        self.number_of_bots = Some(nob);
                    }
                    SnakeSelectionMessage::NumberOfReal(nor) => {
                        if !self.view_model.validate_number_of_real_players(nor) {
                            self.additional_message =
                                Some(format!("Selected an invalid number of real players: {nor}"));
                            return None;
                        }
                        info!("Selected number of real players: {}", nor);
                        if self.number_of_bots.is_some() {
                            let total = self.number_of_bots.unwrap() + nor;
                            let max_allowed = self.view_model.get_max_total_players();
                            if total > max_allowed {
                                self.number_of_bots = Some(max_allowed - nor)
                            }
                        }
                        self.number_of_real_players = Some(nor);
                    }
                    SnakeSelectionMessage::GridSize(gs) => {
                        let nb = match self.number_of_bots {
                            Some(b) => b,
                            None => 0,
                        };
                        let nr = match self.number_of_real_players {
                            Some(r) => r,
                            None => 0,
                        };
                        if !(self.view_model.validate_grid_size(gs)
                            && self.view_model.validate_grid_to_player(gs, nb + nr))
                        {
                            self.additional_message =
                                Some(format!("Selected an invalid grid size: {gs}"));
                            return None;
                        }
                        info!("Selected grid size: {}", gs);
                        self.grid_size = Some(gs);
                    }
                    SnakeSelectionMessage::Submit => {
                        info!("Submitted state: {:#?}", self);
                        if self.selected_bot.is_none()
                            || self.grid_size.is_none()
                            || self.number_of_bots.is_none()
                            || self.number_of_real_players.is_none()
                        {
                            self.additional_message =
                                Some("Please fill in all fields before continuing".to_owned());
                            return None;
                        }
                        let bot_type = self.selected_bot.clone().unwrap();
                        let gs = self.grid_size.unwrap();
                        let nob = self.number_of_bots.unwrap();
                        let nor = self.number_of_real_players.unwrap();
                        if !self.view_model.validate_all(gs, nob, nor) {
                            self.additional_message = Some(
                                "Parameters combination is invalid. Please try changing it."
                                    .to_owned(),
                            );
                            return None;
                        }
                        debug!("Transitioning to snake game");
                        return Some(Message::Snake(SnakeMessage::SnakeGameScreenTransition(
                            SnakeParams {
                                bot_type,
                                grid_size: gs,
                                number_of_real_players: nor,
                                number_of_bots: nob,
                            },
                        )));
                    }
                }
            } else {
                debug!("Received non snake selection message: {:#?}", snake_message);
            }
        } else {
            debug!(
                "Received non snake message in snake selection screen: {:#?}",
                message
            );
        }
        None
    }

    fn view(&self) -> Element<Message> {
        let bot_picker = pick_list(SnakeBotType::VALUES, self.selected_bot.clone(), |bot| {
            Message::Snake(SnakeMessage::SnakeSelectionMessage(
                SnakeSelectionMessage::BotTypeSelected(bot),
            ))
        })
        .placeholder("Select a bot");

        let snor = match self.number_of_real_players {
            Some(r) => r,
            None => 0,
        };

        let num_bot_picker = pick_list(
            Vec::from_iter(0..=(self.view_model.get_max_total_players() - snor)),
            self.number_of_bots,
            |nob| {
                Message::Snake(SnakeMessage::SnakeSelectionMessage(
                    SnakeSelectionMessage::NumberOfBots(nob),
                ))
            },
        )
        .placeholder("Select number of bots");

        let num_player_picker = pick_list(
            Vec::from_iter(0..=self.view_model.get_max_real_players()),
            self.number_of_real_players,
            |nor| {
                Message::Snake(SnakeMessage::SnakeSelectionMessage(
                    SnakeSelectionMessage::NumberOfReal(nor),
                ))
            },
        )
        .placeholder("Select number of players (0, 1, or 2)");

        let total_players = match self.number_of_bots {
            Some(r) => r + snor,
            None => snor,
        };

        let board_size_picker = pick_list(
            Vec::from_iter(
                self.view_model.get_min_grid_size(total_players)
                    ..=self.view_model.get_max_grid_size(),
            ),
            self.grid_size,
            |gs| {
                Message::Snake(SnakeMessage::SnakeSelectionMessage(
                    SnakeSelectionMessage::GridSize(gs),
                ))
            },
        )
        .placeholder("Select board size");

        let submit_button = button(text("Start Game"))
            .on_press(Message::Snake(SnakeMessage::SnakeSelectionMessage(
                SnakeSelectionMessage::Submit,
            )))
            .width(Length::Shrink);
        let home_button = button(text("Back to Home"))
            .on_press(Message::Snake(SnakeMessage::HomeScreenTransition))
            .width(Length::Shrink);

        let content = column![
            text("Choose bot type").size(24),
            bot_picker,
            num_player_picker,
            num_bot_picker,
            board_size_picker,
            submit_button,
            text(if self.additional_message.is_none() {
                "".to_owned()
            } else {
                self.additional_message.clone().unwrap()
            })
            .size(24),
            home_button,
        ]
        .spacing(20)
        .align_x(alignment::Alignment::Center);

        container(content)
            .width(Length::Fill)
            .height(Length::Fill)
            .align_x(alignment::Horizontal::Center)
            .align_y(alignment::Vertical::Center)
            .into()
    }

    fn subscription(&self) -> Subscription<Message> {
        Subscription::none()
    }
}
