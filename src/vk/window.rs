use std::ops::Deref;
pub struct Window (
    Option<winit::window::Window>,
);

impl Window {
    pub fn new() -> Self {
        Self ( None )
    }
}

impl Drop for Window {
    fn drop(&mut self) {
        // Might come in handy some time
    }
}
impl Deref for Window {
    type Target = Option<winit::window::Window>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
