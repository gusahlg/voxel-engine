//! Host tests of the far table, one file per submodule. `support` holds the
//! bodies, views and ray helpers more than one file uses.

// The test files reach the production pack and dip, which take an explicit
// offset table, as `super::pack_table` and `super::horizon_dip`. The
// two-argument wrappers of the same names pass a zero table.
use super::horizon::horizon_dip;
use super::table::pack_table;

mod classify;
mod cones;
mod frozen;
mod horizon;
mod horizon_build;
mod sky_draw;
mod support;
mod table;
mod tiles;
mod view;
