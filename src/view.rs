use iced::{Element, Subscription};

use crate::app::Message;

pub trait View {
    /// Returns an optional message. If there is a message,
    /// switches screen to the variant of the message.
    fn update(&mut self, message: Message) -> Option<Message>;

    fn view(&self) -> Element<Message>;

    fn subscription(&self) -> Subscription<Message>;
}
