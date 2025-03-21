//! Module to handle Snake selection logic.

use std::fmt;

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

/// Wrapper enum to display Option<SnakeBotType>.
#[derive(Debug, Clone, PartialEq)]
pub enum BotOption {
    Bot(SnakeBotType),
    None,
}

impl fmt::Display for BotOption {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BotOption::Bot(bot) => write!(f, "{bot}"),
            BotOption::None => write!(f, "None"),
        }
    }
}

/// Message to propagate information in the [`SnakeSelectionScreen`].
#[derive(Debug, Clone)]
pub enum SnakeSelectionMessage {
    BotTypeSelected(BotOption),
    NumberOfBots(usize),
    NumberOfReal(usize),
    GridSize(usize),
    Submit,
}

/// Struct representing the Snake selection screen.
#[derive(Debug)]
pub struct SnakeSelectionScreen {
    view_model: SnakeSelectionViewModel,
    /// [`SnakeBotType`] to play Snake against.
    selected_bot: Option<BotOption>,
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
    /// Creates a new struct with no selected [`SnakeBotType`].
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
        sss.update(Message::Snake(SnakeMessage::SnakeSelectionMessage(
            SnakeSelectionMessage::BotTypeSelected(match params.bot_type {
                Some(b) => BotOption::Bot(b),
                None => BotOption::None,
            }),
        )));
        sss.update(Message::Snake(SnakeMessage::SnakeSelectionMessage(
            SnakeSelectionMessage::NumberOfBots(params.number_of_bots),
        )));
        sss.update(Message::Snake(SnakeMessage::SnakeSelectionMessage(
            SnakeSelectionMessage::NumberOfReal(params.number_of_real_players),
        )));
        sss.update(Message::Snake(SnakeMessage::SnakeSelectionMessage(
            SnakeSelectionMessage::GridSize(params.grid_size),
        )));
        sss.additional_message = None;
        sss
    }

    pub fn add_message(&mut self, message: String) {
        self.additional_message = Some(message);
    }

    pub fn add_optional_message(&mut self, message: Option<String>) {
        self.additional_message = message;
    }

    fn check_and_reset_grid_size(&mut self) {
        let Some(grid_size) = self.grid_size else {
            return;
        };
        let nb = self.number_of_bots.unwrap_or_default();
        let nr = self.number_of_real_players.unwrap_or_default();
        let total_players = nb + nr;
        if !(self.view_model.validate_grid_size(grid_size)
            && self
                .view_model
                .validate_grid_to_player(grid_size, total_players))
        {
            self.grid_size = Some(self.view_model.get_min_grid_size(total_players));
        }
    }

    fn handle_bot_option(&mut self, bot_type: BotOption) {
        info!("Selected bot: {:#?}", bot_type);
        match bot_type {
            BotOption::Bot(snake_bot_type) => {
                self.selected_bot = Some(BotOption::Bot(snake_bot_type));
            }
            BotOption::None => {
                self.selected_bot = Some(BotOption::None);
                self.number_of_bots = Some(0);
            }
        }
    }

    fn handle_number_of_bots(&mut self, number_of_bots: usize) {
        if !self.view_model.validate_number_of_bots(number_of_bots) {
            self.additional_message = Some(format!(
                "Selected an invalid number of bots: {number_of_bots}"
            ));
            return;
        }
        info!("Selected number of bots: {}", number_of_bots);
        self.number_of_bots = Some(number_of_bots);
        self.check_and_reset_grid_size();
    }

    fn handle_number_of_real_players(&mut self, number_of_real_players: usize) {
        if !self
            .view_model
            .validate_number_of_real_players(number_of_real_players)
        {
            self.additional_message = Some(format!(
                "Selected an invalid number of real players: {number_of_real_players}"
            ));
            return;
        }
        info!(
            "Selected number of real players: {}",
            number_of_real_players
        );
        if self.number_of_bots.is_some() {
            let total = self.number_of_bots.unwrap() + number_of_real_players;
            let max_allowed = self.view_model.get_max_total_players();
            if total > max_allowed {
                self.number_of_bots = Some(max_allowed - number_of_real_players);
            }
        }
        self.number_of_real_players = Some(number_of_real_players);
        self.check_and_reset_grid_size();
    }

    fn handle_grid_size(&mut self, grid_size: usize) {
        let nb = self.number_of_bots.unwrap_or_default();
        let nr = self.number_of_real_players.unwrap_or_default();
        if !(self.view_model.validate_grid_size(grid_size)
            && self.view_model.validate_grid_to_player(grid_size, nb + nr))
        {
            self.additional_message = Some(format!("Selected an invalid grid size: {grid_size}"));
            return;
        }
        info!("Selected grid size: {}", grid_size);
        self.grid_size = Some(grid_size);
    }

    fn handle_submit(&mut self) -> Option<Message> {
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
        let bot_type = match &self.selected_bot {
            Some(b) => match b {
                BotOption::Bot(snake_bot_type) => Some(snake_bot_type.clone()),
                BotOption::None => None,
            },
            None => None,
        };
        let grid_size = self.grid_size.unwrap();
        let number_of_bots = self.number_of_bots.unwrap();
        let number_of_real_players = self.number_of_real_players.unwrap();
        if !self
            .view_model
            .validate_all(grid_size, number_of_bots, number_of_real_players)
        {
            self.additional_message =
                Some("Parameters combination is invalid. Please try changing it.".to_owned());
            return None;
        }
        debug!("Transitioning to snake game");
        Some(Message::Snake(SnakeMessage::SnakeGameScreenTransition(
            SnakeParams {
                bot_type,
                grid_size,
                number_of_real_players,
                number_of_bots,
            },
        )))
    }
}

impl View for SnakeSelectionScreen {
    fn update(&mut self, message: Message) -> Option<Message> {
        debug!("Received message at SnakeSelectionScreen. Evaulating here.");
        let Message::Snake(snake_message) = message else {
            debug!(
                "Received non snake message in snake selection screen: {:#?}",
                message
            );
            return None;
        };
        let SnakeMessage::SnakeSelectionMessage(ssm) = snake_message else {
            debug!("Received non snake selection message: {:#?}", snake_message);
            return None;
        };
        self.additional_message = None;
        match ssm {
            SnakeSelectionMessage::BotTypeSelected(bot_type) => self.handle_bot_option(bot_type),
            SnakeSelectionMessage::NumberOfBots(number_of_bots) => {
                self.handle_number_of_bots(number_of_bots);
            }
            SnakeSelectionMessage::NumberOfReal(number_of_real_players) => {
                self.handle_number_of_real_players(number_of_real_players);
            }
            SnakeSelectionMessage::GridSize(grid_size) => self.handle_grid_size(grid_size),
            SnakeSelectionMessage::Submit => return self.handle_submit(),
        }
        None
    }

    fn view(&self) -> Element<Message> {
        let mut bot_options: Vec<BotOption> = SnakeBotType::VALUES
            .iter()
            .map(|sbt| BotOption::Bot(sbt.clone()))
            .collect();
        bot_options.push(BotOption::None);
        let bot_picker = pick_list(bot_options, self.selected_bot.clone(), |bot| {
            Message::Snake(SnakeMessage::SnakeSelectionMessage(
                SnakeSelectionMessage::BotTypeSelected(bot),
            ))
        })
        .placeholder("Select a bot");

        let snor = self.number_of_real_players.unwrap_or_default();

        let num_bot_picker = pick_list(
            (if self.selected_bot.is_some()
                && matches!(self.selected_bot.as_ref().unwrap(), BotOption::Bot(_))
            {
                0..=(self.view_model.get_max_total_players() - snor)
            } else {
                0..=0
            })
            .collect::<Vec<_>>(),
            self.number_of_bots,
            |number_of_bots| {
                Message::Snake(SnakeMessage::SnakeSelectionMessage(
                    SnakeSelectionMessage::NumberOfBots(number_of_bots),
                ))
            },
        )
        .placeholder("Select number of bots");

        let num_player_picker = pick_list(
            (0..=self.view_model.get_max_real_players()).collect::<Vec<_>>(),
            self.number_of_real_players,
            |number_of_real_players| {
                Message::Snake(SnakeMessage::SnakeSelectionMessage(
                    SnakeSelectionMessage::NumberOfReal(number_of_real_players),
                ))
            },
        )
        .placeholder("Select number of players (0, 1, or 2)");

        let total_players = match self.number_of_bots {
            Some(r) => r + snor,
            None => snor,
        };

        let board_size_picker = pick_list(
            (self.view_model.get_min_grid_size(total_players)
                ..=self.view_model.get_max_grid_size())
                .collect::<Vec<_>>(),
            self.grid_size,
            |grid_size| {
                Message::Snake(SnakeMessage::SnakeSelectionMessage(
                    SnakeSelectionMessage::GridSize(grid_size),
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
            text("Set Game Parameters").size(24),
            bot_picker,
            num_player_picker,
            num_bot_picker,
            board_size_picker,
            submit_button,
            text(if self.additional_message.is_none() {
                String::new()
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
