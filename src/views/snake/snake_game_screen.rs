use std::time::{Duration, Instant};

use iced::{
    keyboard::{self, Key},
    time,
    widget::{button, column, container, row, text, Column, Container, Row},
    Border, Color, Element, Length, Subscription,
};

use crate::{
    app::Message, models::snake::snake_game::SnakeBlock, view::View, view_model::ViewModel,
    view_models::snake::snake_view_model::SnakeViewModel,
};

use super::snake_mediator::SnakeMessage;

#[derive(Clone, Debug)]
pub enum SnakeGameMessage {
    Key(Key),
    Timer(Instant),
    Reset,
}

#[derive(Debug)]
pub struct SnakeGameScreen {
    view_model: SnakeViewModel,
    cell_size: u16,
}

impl SnakeGameScreen {
    #[must_use]
    pub fn new(view_model: SnakeViewModel) -> Self {
        let cell_size = if view_model.get_last_game_board_ref().get_size() < 30 {
            20
        } else {
            16
        };
        Self {
            view_model,
            cell_size,
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

    fn make_container(&self, content: String, color: Color) -> Container<'_, Message> {
        container(
            text(content)
                .color(Color::BLACK)
                .size(self.cell_size * 4 / 5),
        )
        .width(self.cell_size)
        .height(self.cell_size)
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
    }
}

impl View for SnakeGameScreen {
    fn update(&mut self, message: Message) -> Option<Message> {
        self.view_model.update(message)
    }

    fn view(&self) -> Element<Message> {
        let mut grid_view = Column::new();

        let snake_game = self.view_model.get_last_game_board_ref();
        let grid = snake_game.get_grid();
        for grid_row in grid {
            let mut row = Row::new();
            for entry in grid_row {
                let rectangle = match entry {
                    SnakeBlock::Empty => self.make_container(" ".to_owned(), Color::WHITE),
                    SnakeBlock::Apple => {
                        self.make_container(" ".to_owned(), Color::from_rgb(1.0, 0.0, 0.0))
                    }
                    // TODO: Give the snakes head pointing the correct direction by using the whole game info
                    SnakeBlock::SnakeBody(player) => self
                        .make_container(" ".to_owned(), self.get_color_for_player(*player, 0.83)),
                    SnakeBlock::SnakeHead(player) => self.make_container(
                        format!(
                            "{}{}",
                            if self.view_model.get_real_player_indices().contains(player) {
                                "P"
                            } else {
                                "B"
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
        let winner_string = |pindx: usize| {
            if self.view_model.get_real_player_indices().contains(&pindx) {
                format!("Player {}", pindx + 1)
            } else {
                "Bots".to_string()
            }
        };
        if let Some(winner) = snake_game.get_winner() {
            return column!(
                game,
                if self.view_model.get_number_of_players() > 1 {
                    text(format!(
                        "GAME OVER. {} WON!",
                        winner_string(winner)
                    ))
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
        let timer = time::every(Duration::from_millis(
            self.view_model.get_time_between_frames(),
        ))
        .map(SnakeGameMessage::Timer)
        .map(SnakeMessage::SnakeGameMessage)
        .map(Message::Snake);
        Subscription::batch(vec![timer, keyboard])
    }
}
