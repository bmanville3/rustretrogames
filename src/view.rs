use iced::Element;

use crate::app::Message;

pub trait View {
    fn update(&mut self, message: Message) -> Option<Message>;

    fn view(&self) -> Element<Message>;
}
