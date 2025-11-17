use log::debug;
use rustretrogames::app::State;

fn main() {
    std::env::set_var("RUST_LOG", "rustretrogames=debug");
    env_logger::init();
    debug!("Debug on");
    let _ = iced::application("Retro Rust Games", State::update, State::view)
        .window_size(iced::Size::new(1000.0, 800.0))
        .subscription(State::subscription)
        .run();
}
