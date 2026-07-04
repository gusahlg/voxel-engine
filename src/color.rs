//! RGBA8 color type with raylib-compatible named constants.

use bytemuck::{Pod, Zeroable};

/// An RGBA color with 8 bits per channel. Layout-compatible with GPU vertex
/// data (`repr(C)`, Pod). Constant names and values match raylib exactly so
/// ported game code keeps its colors bit-for-bit.
#[repr(C)]
#[derive(Clone, Copy, PartialEq, Eq, Debug, Pod, Zeroable)]
pub struct Color {
    pub r: u8,
    pub g: u8,
    pub b: u8,
    pub a: u8,
}

impl Color {
    pub const fn new(r: u8, g: u8, b: u8, a: u8) -> Self {
        Self { r, g, b, a }
    }

    /// Opaque color (alpha 255).
    pub const fn rgb(r: u8, g: u8, b: u8) -> Self {
        Self { r, g, b, a: 255 }
    }

    /// Returns the color with its alpha multiplied by `alpha`, clamped to [0, 1].
    pub fn fade(self, alpha: f32) -> Self {
        let a = (self.a as f32 * alpha.clamp(0.0, 1.0)) as u8;
        Self { a, ..self }
    }

    pub const LIGHTGRAY: Self = Self::rgb(200, 200, 200);
    pub const GRAY: Self = Self::rgb(130, 130, 130);
    pub const DARKGRAY: Self = Self::rgb(80, 80, 80);
    pub const YELLOW: Self = Self::rgb(253, 249, 0);
    pub const GOLD: Self = Self::rgb(255, 203, 0);
    pub const ORANGE: Self = Self::rgb(255, 161, 0);
    pub const PINK: Self = Self::rgb(255, 109, 194);
    pub const RED: Self = Self::rgb(230, 41, 55);
    pub const MAROON: Self = Self::rgb(190, 33, 55);
    pub const GREEN: Self = Self::rgb(0, 228, 48);
    pub const LIME: Self = Self::rgb(0, 158, 47);
    pub const DARKGREEN: Self = Self::rgb(0, 117, 44);
    pub const SKYBLUE: Self = Self::rgb(102, 191, 255);
    pub const BLUE: Self = Self::rgb(0, 121, 241);
    pub const DARKBLUE: Self = Self::rgb(0, 82, 172);
    pub const PURPLE: Self = Self::rgb(200, 122, 255);
    pub const VIOLET: Self = Self::rgb(135, 60, 190);
    pub const DARKPURPLE: Self = Self::rgb(112, 31, 126);
    pub const BEIGE: Self = Self::rgb(211, 176, 131);
    pub const BROWN: Self = Self::rgb(127, 106, 79);
    pub const DARKBROWN: Self = Self::rgb(76, 63, 47);
    pub const WHITE: Self = Self::rgb(255, 255, 255);
    pub const BLACK: Self = Self::rgb(0, 0, 0);
    pub const BLANK: Self = Self::new(0, 0, 0, 0);
    pub const MAGENTA: Self = Self::rgb(255, 0, 255);
    pub const RAYWHITE: Self = Self::rgb(245, 245, 245);
    pub const SALMON: Self = Self::rgb(250, 128, 114);
}
