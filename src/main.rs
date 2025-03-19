use log::debug;
use rustretrogames::app::State;

#[tokio::main]
async fn main() {
    std::env::set_var("RUST_LOG", "rustretrogames=debug");
    env_logger::init();
    debug!("Debug on");
    let _ = iced::application("Retro Rust Games", State::update, State::view)
        .subscription(State::subscription)
        .run();
}
