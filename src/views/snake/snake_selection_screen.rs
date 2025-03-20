use iced::{
    alignment,
    widget::{button, column, container, pick_list, text},
    Element, Length, Subscription,
};
use log::debug;

use crate::{app::Message, models::snake::snake_bot::SnakeBotType, view::View};

use super::snake_mediator::SnakeMessage;

#[derive(Debug, Clone)]
pub enum SnakeSelectionMessage {
    BotTypeSelected(SnakeBotType),
    Submit,
}

#[derive(Debug)]
pub struct SnakeSelectionScreen {
    selected_bot: Option<SnakeBotType>,
}

impl Default for SnakeSelectionScreen {
    fn default() -> Self {
        Self::new()
    }
}

impl SnakeSelectionScreen {
    #[must_use]
    pub fn new() -> Self {
        Self { selected_bot: None }
    }
}

impl View for SnakeSelectionScreen {
    fn update(&mut self, message: Message) -> Option<Message> {
        if let Message::Snake(snake_message) = message {
            if let SnakeMessage::SnakeSelectionMessage(ssm) = snake_message {
                match ssm {
                    SnakeSelectionMessage::BotTypeSelected(bot_type) => {
                        debug!("Selected bot: {}", bot_type);
                        self.selected_bot = Some(bot_type);
                    }
                    SnakeSelectionMessage::Submit => {
                        debug!("Submitted selected bot: {:#?}", self.selected_bot);
                        if let Some(bot) = self.selected_bot.clone() {
                            debug!("Transitioning to snake game");
                            return Some(Message::Snake(SnakeMessage::SnakeGameScreenTransition(
                                bot,
                            )));
                        }
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
            submit_button,
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
