pub trait Environment
where
    Self::State: Clone + Into<Vec<f32>>,
    Self::Action: Clone,
{
    type State;
    type Action;

    fn reset(&mut self) -> Self::State;
    fn step(&mut self, action: &Self::Action) -> (Self::State, f32, bool);
    fn get_action_mask(&self) -> Vec<bool>;

    fn all_actions() -> Vec<Self::Action>;
    fn action_to_index(action: &Self::Action) -> usize;
}
