pub trait Environment
where
    Self::State: Clone,
    Self::Action: Clone,
{
    type State;
    type Action;

    fn reset(&mut self) -> Self::State;
    fn step(&mut self, action: &Self::Action) -> (Self::State, f32, bool);
    fn available_actions(&self) -> Vec<Self::Action>;
}
