use std::time::{Duration, Instant};

use iced::{
    keyboard::{self, Key},
    time,
    widget::{button, column, container, row, text, Column, Row},
    Border, Color, Element, Length, Subscription,
};
use log::debug;

use crate::{
    app::Message,
    models::snake::snake_game::SnakeBlock,
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
    Reset(bool),
}

#[derive(Debug)]
pub struct SnakeGameScreen {
    view_model: SnakeViewModel,
    sub_key: usize,
    needs_reset: bool,
}

impl SnakeGameScreen {
    #[must_use]
    pub fn new(view_model: SnakeViewModel, sub_key: usize) -> Self {
        Self {
            view_model,
            sub_key,
            needs_reset: true,
        }
    }

    fn get_color_for_player(&self, player: usize, alpha: f32) -> Color {
        #[allow(clippy::cast_precision_loss)]
        let x = player as f32 / self.view_model.get_number_of_players() as f32;
        assert!((0.0..=1.0).contains(&x));
        let xs = x.powi(2);
        let red = 1.0 - xs;
        let green = 0.5 * xs + 0.5;
        let mut blue = 5.0 * xs - 5.0 * x + 1.0;
        if blue <= 0.1 {
            blue = 0.1;
        }
        Color::from_rgba(red, green, blue, alpha)
    }
}

impl View for SnakeGameScreen {
    fn update(&mut self, message: Message) -> Option<Message> {
        debug!("Received message at SnakeGameScreen. Evaluating here.");
        if let Message::Snake(SnakeMessage::SnakeGameMessage(SnakeGameMessage::Reset(r))) = message
        {
            debug!("Turning need reset to {r}");
            self.needs_reset = r;
        }
        self.view_model.update(message)
    }

    fn view(&self) -> Element<Message> {
        let mut grid_view = Column::new();
        let cell_size = if self.view_model.get_game_ref().get_size() < 30 {
            20
        } else {
            16
        };

        let make_container = |content: String, color: Color| {
            container(text(content).color(Color::BLACK).size(cell_size * 4 / 5))
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

        let game = self.view_model.get_game_ref();
        let grid = game.get_grid();
        for grid_row in grid {
            let mut row = Row::new();
            for entry in grid_row {
                let rectangle = match entry {
                    SnakeBlock::Empty => make_container(" ".to_owned(), Color::WHITE),
                    SnakeBlock::Apple => {
                        make_container(" ".to_owned(), Color::from_rgb(1.0, 0.0, 0.0))
                    }
                    // TODO: Give the snakes head pointing the correct direction by using the whole game info
                    SnakeBlock::SnakeBody(player) => {
                        make_container(" ".to_owned(), self.get_color_for_player(*player, 0.83))
                    }
                    SnakeBlock::SnakeHead(player) => make_container(
                        format!(
                            "{}{}",
                            if self.view_model.get_players()[*player].is_bot {
                                "B"
                            } else {
                                "P"
                            },
                            player + 1
                        ),
                        self.get_color_for_player(*player, 1.0),
                    ),
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
            .on_press(Message::Snake(
                SnakeMessage::SnakeSelectionScreenTransition((
                    Some(self.view_model.get_params()),
                    None,
                )),
            ))
            .width(160)
            .height(40);
        let restart_button = button(text("Restart"))
            .on_press(Message::Snake(SnakeMessage::SnakeGameMessage(
                SnakeGameMessage::Reset(true),
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
        if winner.is_some() {
            return column!(
                game,
                if self.view_model.get_number_of_players() > 1 {
                    text(format!("GAME OVER. {} WON!", winner.unwrap().get_name()))
                } else {
                    text("GAME OVER")
                }
            )
            .align_x(iced::alignment::Horizontal::Center)
            .into();
        } else if self.view_model.real_players_lost() {
            return column!(game, text("ALL THE REAL PLAYERS LOST!"))
                .align_x(iced::alignment::Horizontal::Center)
                .into();
        }
        game.into()
    }

    fn subscription(&self) -> Subscription<Message> {
        let keyboard = keyboard::on_key_press(|key, _| {
            Some(Message::Snake(SnakeMessage::SnakeGameMessage(
                SnakeGameMessage::Key(key),
            )))
        });
        if self.view_model.game_over() {
            return keyboard;
        }
        if self.needs_reset {
            // if there is a winner drop all subscriptions (as the game is over)
            debug!("Reseting Subscription");
            return time::every(Duration::from_millis(10))
                .map(|_| SnakeGameMessage::Reset(false))
                .map(SnakeMessage::SnakeGameMessage)
                .map(Message::Snake);
        }
        let timer = time::every(Duration::from_millis(
            self.view_model.get_time_between_frames(),
        ))
        .map(SnakeGameMessage::Timer)
        .map(SnakeMessage::SnakeGameMessage)
        .map(Message::Snake);
        let mut sub = Subscription::batch(vec![timer, keyboard]);
        for player in self.view_model.get_players() {
            if !player.is_bot {
                continue;
            }
            let bot = Subscription::run_with_id(
                self.sub_key.wrapping_add(player.player_id + 1),
                self.view_model.make_bot_thread(player.player_id),
            );
            sub = Subscription::batch(vec![sub, bot]);
        }
        sub
    }
}
