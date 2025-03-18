use rustretrogames::app::State;
fn main() {
    let _ = iced::run("Retro Rust Games", State::update, State::view);
}
