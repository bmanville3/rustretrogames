use std::time::{Duration, Instant};

use iced::{
    keyboard::{self, Key},
    time,
    widget::{button, column, container, row, text, Column, Row},
    Border, Color, Element, Length, Subscription,
};

use crate::{
    app::Message,
    models::snake::snake_model::SnakeBlock,
    view::View,
    view_model::ViewModel,
    view_models::snake::snake_view_model::{ChannelMessage, SnakeViewModel},
};

use super::snake_mediator::SnakeMessage;

#[derive(Clone, Debug)]
pub enum SnakeGameMessage {
    ChannelMessage(ChannelMessage),
    Key(Key),
    Timer(Instant),
    Reset,
}

#[derive(Debug)]
pub struct SnakeGameScreen {
    view_model: SnakeViewModel,
    sub_key: u64,
}

impl SnakeGameScreen {
    #[must_use]
    pub fn new(view_model: SnakeViewModel, sub_key: u64) -> Self {
        Self {
            view_model,
            sub_key,
        }
    }
}

impl View for SnakeGameScreen {
    fn update(&mut self, message: Message) -> Option<Message> {
        self.view_model.update(message)
    }

    fn view(&self) -> Element<Message> {
        let mut grid_view = Column::new();
        let cell_size = 20;

        let make_container = |color: Color| {
            container(text(" ").color(color)) // Empty text to preserve size
                .width(cell_size)
                .height(cell_size)
                .style(move |_: &_| container::Style {
                    // TODO Fix this
                    border: Border {
                        color: Color::from_rgba(0.0, 0.0, 0.0, 0.1),
                        width: 1.0,
                        ..Default::default()
                    },
                    background: Some(color.into()),
                    ..container::Style::default()
                })
        };

        let grid = self.view_model.get_ref_backing_grid();
        for grid_row in grid {
            let mut row = Row::new();
            for entry in grid_row {
                let rectangle = match entry {
                    SnakeBlock::EMPTY => make_container(Color::WHITE),
                    SnakeBlock::APPLE => make_container(Color::from_rgb(1.0, 0.0, 0.0)),
                    SnakeBlock::PLAYERONE => make_container(Color::from_rgba(0.0, 1.0, 0.0, 0.8)),
                    SnakeBlock::PLAYERTWO => make_container(Color::from_rgba(0.0, 0.0, 1.0, 0.8)),
                    SnakeBlock::HEADONE => make_container(Color::from_rgb(0.0, 1.0, 0.0)),
                    SnakeBlock::HEADTWO => make_container(Color::from_rgb(0.0, 0.0, 1.0)),
                };

                row = row.push(rectangle);
            }
            grid_view = grid_view.push(row);
        }

        let home_button = button(text("Back to Home"))
            .on_press(Message::Snake(SnakeMessage::HomeScreenTransition))
            .width(160)
            .height(40);
        let back_button = button(text("Go back"))
            .on_press(Message::Snake(SnakeMessage::SnakeSelectionScreenTransition))
            .width(160)
            .height(40);
        let restart_button = button(text("Restart"))
            .on_press(Message::Snake(SnakeMessage::SnakeGameMessage(
                SnakeGameMessage::Reset,
            )))
            .width(80)
            .height(40);

        let game = container(
            column![
                row![home_button, back_button, restart_button].spacing(10), // Keep the home button at the top left
                grid_view, // Below it, display the game grid
            ]
            .spacing(10),
        )
        .width(Length::Fill)
        .height(Length::Fill)
        .align_x(iced::alignment::Horizontal::Center)
        .align_y(iced::alignment::Vertical::Center);
        let winner = self.view_model.get_winner();
        if winner != 0 {
            return column!(
                game,
                text(format!(
                    "GAME OVER. YOU {}!",
                    if winner == 2 { "LOST" } else { "WON" }
                ))
            )
            .align_x(iced::alignment::Horizontal::Center)
            .into();
        }
        game.into()
    }

    fn subscription(&self) -> Subscription<Message> {
        let timer = time::every(Duration::from_millis(
            self.view_model.get_time_between_frames(),
        ))
        .map(SnakeGameMessage::Timer)
        .map(SnakeMessage::SnakeGameMessage)
        .map(Message::Snake);
        let keyboard = keyboard::on_key_press(|key, _| {
            Some(Message::Snake(SnakeMessage::SnakeGameMessage(
                SnakeGameMessage::Key(key),
            )))
        });
        let bot = Subscription::run_with_id(self.sub_key, self.view_model.make_bot_thread());
        Subscription::batch(vec![timer, keyboard, bot])
    }
}
