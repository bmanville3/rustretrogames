//! The [`ViewModel`] trait for the MVVM architecture.

use crate::app::Message;

/// Trait containing methods for `ViewModel` modules in the MVVM architecture.
pub trait ViewModel {
    /// Updates the [`crate::view::View`].
    fn update(&mut self, message: Message) -> Option<Message>;
}
