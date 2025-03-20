use crate::app::Message;

pub trait ViewModel {
    fn update(&mut self, message: Message) -> Option<Message>;
}
