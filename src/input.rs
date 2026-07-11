//! Raylib-style polling input facade fed by winit events.
//!
//! The engine forwards winit window/device events here as they arrive, and
//! calls [`InputState::begin_frame`] once per frame before the game callback.
//! The game then polls with `is_key_down` / `is_key_pressed` / etc.
//!
//! Raylib parity contracts:
//! - `is_key_pressed` is true exactly on the transition frame and is stable
//!   within that frame (set-based, no consumption on query).
//! - The char queue is layout+shift-aware (`KeyEvent.text`), drainable
//!   mid-frame via `get_char_pressed`, and reset every frame like raylib's.
//! - `mouse_delta` accumulates raw device motion and keeps working while the
//!   cursor is locked/hidden.

use std::cell::RefCell;
use std::collections::{HashSet, VecDeque};

use winit::event::{DeviceEvent, WindowEvent};
use winit::keyboard::{KeyCode, PhysicalKey};

/// Physical keyboard keys the engine exposes (raylib-style names).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Key {
    A,
    B,
    C,
    D,
    E,
    F,
    G,
    H,
    I,
    J,
    K,
    L,
    M,
    N,
    O,
    P,
    Q,
    R,
    S,
    T,
    U,
    V,
    W,
    X,
    Y,
    Z,
    Num0,
    Num1,
    Num2,
    Num3,
    Num4,
    Num5,
    Num6,
    Num7,
    Num8,
    Num9,
    Space,
    Enter,
    Escape,
    Backspace,
    Tab,
    LeftShift,
    RightShift,
    LeftControl,
    RightControl,
    LeftAlt,
    Slash,
    Period,
    Comma,
    Minus,
    Equal,
    Apostrophe,
    Semicolon,
    Up,
    Down,
    Left,
    Right,
    Home,
    End,
    Delete,
    F1,
    F2,
    F3,
}

/// Mouse buttons the engine exposes. Other buttons are ignored.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum MouseButton {
    Left,
    Right,
    Middle,
}

/// Polled input state for one window.
///
/// Single-threaded by design: the char queue lives in a `RefCell` so
/// `get_char_pressed(&self)` can drain it through a shared reference
/// (matching raylib's `GetCharPressed`), which makes this type `!Sync`.
pub struct InputState {
    keys_down: HashSet<Key>,
    keys_pressed: HashSet<Key>,
    mouse_down: HashSet<MouseButton>,
    mouse_pressed: HashSet<MouseButton>,
    mouse_delta: (f64, f64),
    /// Vertical scroll accumulated since `begin_frame`, in line units (a mouse
    /// notch is ~1.0; pixel-delta devices are normalised to the same scale).
    scroll_delta: f32,
    chars: RefCell<VecDeque<char>>,
}

impl InputState {
    pub fn new() -> Self {
        Self {
            keys_down: HashSet::new(),
            keys_pressed: HashSet::new(),
            mouse_down: HashSet::new(),
            mouse_pressed: HashSet::new(),
            mouse_delta: (0.0, 0.0),
            scroll_delta: 0.0,
            chars: RefCell::new(VecDeque::new()),
        }
    }

    /// Called once per frame, before the game callback. Clears the
    /// pressed-edge sets, the mouse delta, and any undrained chars (raylib
    /// resets its char queue every poll). Held-key state persists.
    pub fn begin_frame(&mut self) {
        self.keys_pressed.clear();
        self.mouse_pressed.clear();
        self.mouse_delta = (0.0, 0.0);
        self.scroll_delta = 0.0;
        self.chars.borrow_mut().clear();
    }

    /// Feed a winit window event (keyboard, mouse buttons, focus).
    pub fn on_window_event(&mut self, event: &WindowEvent) {
        match event {
            WindowEvent::KeyboardInput { event, .. } => {
                if let PhysicalKey::Code(code) = event.physical_key
                    && let Some(key) = map_key(code)
                {
                    self.key_event(key, event.state.is_pressed(), event.repeat);
                }
                // Text is layout-aware and fires on repeats too, independent
                // of whether we map the physical key.
                if event.state.is_pressed()
                    && let Some(text) = &event.text
                {
                    self.text_input(text.as_str());
                }
            }
            WindowEvent::MouseInput { state, button, .. } => {
                let b = match button {
                    winit::event::MouseButton::Left => MouseButton::Left,
                    winit::event::MouseButton::Right => MouseButton::Right,
                    winit::event::MouseButton::Middle => MouseButton::Middle,
                    _ => return,
                };
                self.mouse_button_event(b, state.is_pressed());
            }
            WindowEvent::MouseWheel { delta, .. } => {
                // Normalise both wheel encodings to line units: notched wheels
                // report lines directly; trackpads/precision wheels report
                // pixels, which we scale down to a comparable notch size.
                let lines = match delta {
                    winit::event::MouseScrollDelta::LineDelta(_, y) => *y,
                    winit::event::MouseScrollDelta::PixelDelta(p) => p.y as f32 / 50.0,
                };
                self.scroll_delta += lines;
            }
            WindowEvent::Focused(false) => self.focus_lost(),
            _ => {}
        }
    }

    /// Feed a winit device event (raw mouse motion).
    pub fn on_device_event(&mut self, event: &DeviceEvent) {
        if let DeviceEvent::MouseMotion { delta } = event {
            self.mouse_motion(delta.0, delta.1);
        }
    }

    pub fn is_key_down(&self, k: Key) -> bool {
        self.keys_down.contains(&k)
    }

    /// True only on the frame the key transitioned to pressed. Stable within
    /// a frame: every call site sees the same answer.
    pub fn is_key_pressed(&self, k: Key) -> bool {
        self.keys_pressed.contains(&k)
    }

    /// Pops the next typed character (FIFO), or `None` when drained.
    pub fn get_char_pressed(&self) -> Option<char> {
        self.chars.borrow_mut().pop_front()
    }

    /// Raw relative mouse motion accumulated since `begin_frame`.
    pub fn mouse_delta(&self) -> glam::Vec2 {
        glam::Vec2::new(self.mouse_delta.0 as f32, self.mouse_delta.1 as f32)
    }

    /// Vertical scroll accumulated since `begin_frame`, positive when scrolling
    /// up/away. In line units (a mouse notch is ~1.0).
    pub fn mouse_wheel(&self) -> f32 {
        self.scroll_delta
    }

    pub fn is_mouse_button_pressed(&self, b: MouseButton) -> bool {
        self.mouse_pressed.contains(&b)
    }

    pub fn is_mouse_button_down(&self, b: MouseButton) -> bool {
        self.mouse_down.contains(&b)
    }

    // ---- internal event handlers (unit-testable; winit events can't be
    // constructed by hand) ----

    fn key_event(&mut self, key: Key, pressed: bool, repeat: bool) {
        if pressed {
            if !repeat {
                self.keys_down.insert(key);
                self.keys_pressed.insert(key);
            }
            // Repeats touch neither set: the key is already held and a
            // repeat is not a new press edge.
        } else {
            self.keys_down.remove(&key);
        }
    }

    fn text_input(&mut self, s: &str) {
        let mut chars = self.chars.borrow_mut();
        for c in s.chars() {
            if !c.is_control() {
                chars.push_back(c);
            }
        }
    }

    fn mouse_button_event(&mut self, b: MouseButton, pressed: bool) {
        if pressed {
            self.mouse_down.insert(b);
            self.mouse_pressed.insert(b);
        } else {
            self.mouse_down.remove(&b);
        }
    }

    fn mouse_motion(&mut self, dx: f64, dy: f64) {
        self.mouse_delta.0 += dx;
        self.mouse_delta.1 += dy;
    }

    /// Focus loss releases everything held, so keys can't stick down while
    /// the window can no longer see their release events.
    fn focus_lost(&mut self) {
        self.keys_down.clear();
        self.mouse_down.clear();
    }
}

/// Maps a winit physical key code to an engine `Key`. Unmapped keys are
/// ignored (they still contribute text input).
fn map_key(code: KeyCode) -> Option<Key> {
    Some(match code {
        KeyCode::KeyA => Key::A,
        KeyCode::KeyB => Key::B,
        KeyCode::KeyC => Key::C,
        KeyCode::KeyD => Key::D,
        KeyCode::KeyE => Key::E,
        KeyCode::KeyF => Key::F,
        KeyCode::KeyG => Key::G,
        KeyCode::KeyH => Key::H,
        KeyCode::KeyI => Key::I,
        KeyCode::KeyJ => Key::J,
        KeyCode::KeyK => Key::K,
        KeyCode::KeyL => Key::L,
        KeyCode::KeyM => Key::M,
        KeyCode::KeyN => Key::N,
        KeyCode::KeyO => Key::O,
        KeyCode::KeyP => Key::P,
        KeyCode::KeyQ => Key::Q,
        KeyCode::KeyR => Key::R,
        KeyCode::KeyS => Key::S,
        KeyCode::KeyT => Key::T,
        KeyCode::KeyU => Key::U,
        KeyCode::KeyV => Key::V,
        KeyCode::KeyW => Key::W,
        KeyCode::KeyX => Key::X,
        KeyCode::KeyY => Key::Y,
        KeyCode::KeyZ => Key::Z,
        KeyCode::Digit0 => Key::Num0,
        KeyCode::Digit1 => Key::Num1,
        KeyCode::Digit2 => Key::Num2,
        KeyCode::Digit3 => Key::Num3,
        KeyCode::Digit4 => Key::Num4,
        KeyCode::Digit5 => Key::Num5,
        KeyCode::Digit6 => Key::Num6,
        KeyCode::Digit7 => Key::Num7,
        KeyCode::Digit8 => Key::Num8,
        KeyCode::Digit9 => Key::Num9,
        KeyCode::Space => Key::Space,
        KeyCode::Enter => Key::Enter,
        KeyCode::Escape => Key::Escape,
        KeyCode::Backspace => Key::Backspace,
        KeyCode::Tab => Key::Tab,
        KeyCode::ShiftLeft => Key::LeftShift,
        KeyCode::ShiftRight => Key::RightShift,
        KeyCode::ControlLeft => Key::LeftControl,
        KeyCode::ControlRight => Key::RightControl,
        KeyCode::AltLeft => Key::LeftAlt,
        KeyCode::Slash => Key::Slash,
        KeyCode::Period => Key::Period,
        KeyCode::Comma => Key::Comma,
        KeyCode::Minus => Key::Minus,
        KeyCode::Equal => Key::Equal,
        KeyCode::Quote => Key::Apostrophe,
        KeyCode::Semicolon => Key::Semicolon,
        KeyCode::ArrowUp => Key::Up,
        KeyCode::ArrowDown => Key::Down,
        KeyCode::ArrowLeft => Key::Left,
        KeyCode::ArrowRight => Key::Right,
        KeyCode::Home => Key::Home,
        KeyCode::End => Key::End,
        KeyCode::Delete => Key::Delete,
        KeyCode::F1 => Key::F1,
        KeyCode::F2 => Key::F2,
        KeyCode::F3 => Key::F3,
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn press_edge_lasts_one_frame_held_persists() {
        let mut input = InputState::new();
        input.begin_frame();
        input.key_event(Key::W, true, false);
        assert!(input.is_key_pressed(Key::W));
        // Stable within the frame: repeated queries agree.
        assert!(input.is_key_pressed(Key::W));
        assert!(input.is_key_down(Key::W));

        input.begin_frame();
        assert!(!input.is_key_pressed(Key::W), "edge must clear next frame");
        assert!(input.is_key_down(Key::W), "held state must persist");

        input.key_event(Key::W, false, false);
        assert!(!input.is_key_down(Key::W));
    }

    #[test]
    fn repeat_makes_no_edge_but_appends_chars() {
        let mut input = InputState::new();
        input.begin_frame();
        input.key_event(Key::W, true, false);
        input.text_input("w");

        input.begin_frame();
        input.key_event(Key::W, true, true);
        input.text_input("w");
        assert!(!input.is_key_pressed(Key::W), "repeat is not a press edge");
        assert!(input.is_key_down(Key::W));
        assert_eq!(input.get_char_pressed(), Some('w'), "repeat still types");
        assert_eq!(input.get_char_pressed(), None);
    }

    #[test]
    fn char_queue_is_fifo_and_reset_each_frame() {
        let mut input = InputState::new();
        input.begin_frame();
        input.text_input("ab");
        input.text_input("c");
        assert_eq!(input.get_char_pressed(), Some('a'));
        assert_eq!(input.get_char_pressed(), Some('b'));

        // Undrained 'c' is discarded at the frame boundary (raylib parity).
        input.begin_frame();
        assert_eq!(input.get_char_pressed(), None);
    }

    #[test]
    fn control_chars_are_filtered() {
        let mut input = InputState::new();
        input.begin_frame();
        input.text_input("\r");
        input.text_input("\u{8}");
        input.text_input("\ta\u{1b}");
        assert_eq!(input.get_char_pressed(), Some('a'));
        assert_eq!(input.get_char_pressed(), None);
    }

    #[test]
    fn focus_loss_clears_held_keys_and_buttons() {
        let mut input = InputState::new();
        input.begin_frame();
        input.key_event(Key::W, true, false);
        input.key_event(Key::LeftShift, true, false);
        input.mouse_button_event(MouseButton::Left, true);

        input.focus_lost();
        assert!(!input.is_key_down(Key::W));
        assert!(!input.is_key_down(Key::LeftShift));
        assert!(!input.is_mouse_button_down(MouseButton::Left));
    }

    #[test]
    fn mouse_delta_accumulates_and_resets() {
        let mut input = InputState::new();
        input.begin_frame();
        input.mouse_motion(1.5, 2.0);
        input.mouse_motion(0.5, -1.0);
        assert_eq!(input.mouse_delta(), glam::Vec2::new(2.0, 1.0));

        input.begin_frame();
        assert_eq!(input.mouse_delta(), glam::Vec2::ZERO);
    }

    #[test]
    fn mouse_button_edge_vs_held() {
        let mut input = InputState::new();
        input.begin_frame();
        input.mouse_button_event(MouseButton::Right, true);
        assert!(input.is_mouse_button_pressed(MouseButton::Right));
        assert!(input.is_mouse_button_down(MouseButton::Right));

        input.begin_frame();
        assert!(!input.is_mouse_button_pressed(MouseButton::Right));
        assert!(input.is_mouse_button_down(MouseButton::Right));

        input.mouse_button_event(MouseButton::Right, false);
        assert!(!input.is_mouse_button_down(MouseButton::Right));
    }
}
