use iced::{Element, Subscription};

use crate::app::Message;

pub trait View {
    fn update(&mut self, message: Message) -> Option<Message>;

    fn view(&self) -> Element<Message>;

    fn subscription(&self) -> Subscription<Message>;
}
