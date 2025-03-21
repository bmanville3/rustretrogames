//! The [View] trait for the MVVM architecture.

use iced::{Element, Subscription};

use crate::app::Message;

/// Trait containing methods for View modules in the MVVM architecture.
pub trait View {
    /// Updates the view.
    ///
    /// Returns an optional [Message]. If there is a message,
    /// switches screen to the variant of the message.
    fn update(&mut self, message: Message) -> Option<Message>;

    /// Displays the view.
    fn view(&self) -> Element<Message>;

    /// Returns the [Subscription] of the view.
    fn subscription(&self) -> Subscription<Message>;
}
