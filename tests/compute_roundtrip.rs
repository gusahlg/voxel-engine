//! GPU compute roundtrip: register the example integer-hash shader, submit
//! 64 jobs of 4096 u32 outputs with staged inputs, poll until they return,
//! and check bit-identical CPU wrapping-hash results.
//!
//! Skips when no windowing display is present (CI / headless). Requires a
//! Vulkan 1.3 device, like a live `cargo run --bin demo`.

use std::sync::mpsc;

use voxel_engine::{Color, ComputeJob, Config, Engine, EngineError};

fn hash32(mut x: u32) -> u32 {
    x ^= x >> 16;
    x = x.wrapping_mul(0x7feb352d);
    x ^= x >> 15;
    x = x.wrapping_mul(0x846ca68b);
    x ^= x >> 16;
    x
}

fn has_display() -> bool {
    std::env::var_os("DISPLAY").is_some() || std::env::var_os("WAYLAND_DISPLAY").is_some()
}

const JOBS: u32 = 64;
const COUNT: u32 = 4096;
const OUTPUT_BYTES: u32 = COUNT * 4;

#[test]
fn compute_roundtrip() {
    if !has_display() {
        eprintln!("skipping compute_roundtrip: no DISPLAY/WAYLAND_DISPLAY");
        return;
    }
    // Winit refuses EventLoop::new off the process main thread (cargo test).
    unsafe { std::env::set_var("VOXEL_EVENTLOOP_ANY_THREAD", "1") };

    let (tx, rx) = mpsc::channel::<Result<(), String>>();
    let cfg = Config {
        title: "compute_roundtrip".into(),
        width: 64,
        height: 64,
        vsync: false,
        target_fps: 0,
        resizable: false,
        ..Config::default()
    };

    let mut kind = None;
    let mut expected: Vec<Vec<u32>> = Vec::new();
    let mut ids = Vec::new();
    let mut got: Vec<(voxel_engine::JobId, Box<[u8]>)> = Vec::new();
    let mut frames = 0u32;

    voxel_engine::run(cfg, move |eng: &mut Engine| {
        frames += 1;
        if kind.is_none() {
            let k = match eng.register_compute(&Engine::example_compute_desc()) {
                Ok(k) => k,
                Err(EngineError::NoCompute) => {
                    let _ = tx.send(Err("register_compute: NoCompute".into()));
                    return false;
                }
                Err(e) => {
                    let _ = tx.send(Err(format!("register_compute: {e}")));
                    return false;
                }
            };
            let stager = eng.compute_stager();
            let queue = eng.compute_queue();
            for job_i in 0..JOBS {
                let mut input = match stager.acquire(OUTPUT_BYTES as usize) {
                    Some(i) => i,
                    None => {
                        let _ = tx.send(Err("input ring full at acquire".into()));
                        return false;
                    }
                };
                {
                    let bytes = input.bytes();
                    for i in 0..COUNT {
                        let b = i.to_le_bytes();
                        let o = (i * 4) as usize;
                        bytes[o..o + 4].copy_from_slice(&b);
                    }
                }
                let seed = job_i;
                let mut push = [0u8; 8];
                push[..4].copy_from_slice(&seed.to_le_bytes());
                push[4..].copy_from_slice(&COUNT.to_le_bytes());
                match queue.submit(ComputeJob {
                    kind: k,
                    push: &push,
                    inputs: [Some(input), None],
                    output_bytes: OUTPUT_BYTES,
                    dispatch: [64, 1, 1],
                }) {
                    Ok(id) => ids.push(id),
                    Err(e) => {
                        let _ = tx.send(Err(format!("submit {job_i}: {e}")));
                        return false;
                    }
                }
                expected.push((0..COUNT).map(|i| hash32(i ^ seed)).collect());
            }
            kind = Some(k);
            assert_eq!(eng.compute_pending(), JOBS as usize);
        }

        got.extend(eng.poll_compute());
        if got.len() >= JOBS as usize {
            if let Err(e) = check(&ids, &got, &expected) {
                let _ = tx.send(Err(e));
            } else {
                let _ = tx.send(Ok(()));
            }
            return false;
        }
        if frames > 1000 {
            let _ = tx.send(Err(format!(
                "timeout after {frames} frames: {} of {JOBS} jobs",
                got.len()
            )));
            return false;
        }
        // Pump the render thread so queued jobs are recorded and submitted.
        let _frame = eng.begin_frame(Color::BLACK.to_linear());
        true
    });

    match rx.recv() {
        Ok(Ok(())) => {}
        Ok(Err(e)) => panic!("{e}"),
        Err(_) => panic!("engine exited without a compute_roundtrip result"),
    }
}

fn check(
    ids: &[voxel_engine::JobId],
    got: &[(voxel_engine::JobId, Box<[u8]>)],
    expected: &[Vec<u32>],
) -> Result<(), String> {
    if got.len() != ids.len() {
        return Err(format!("got {} results, expected {}", got.len(), ids.len()));
    }
    for (i, ((id, bytes), exp)) in got.iter().zip(expected.iter()).enumerate() {
        if *id != ids[i] {
            return Err(format!("job {i}: id {:?} != {:?}", id, ids[i]));
        }
        if bytes.len() != OUTPUT_BYTES as usize {
            return Err(format!(
                "job {i}: {} bytes, expected {OUTPUT_BYTES}",
                bytes.len()
            ));
        }
        for (j, want) in exp.iter().enumerate() {
            let o = j * 4;
            let got_u = u32::from_le_bytes(bytes[o..o + 4].try_into().unwrap());
            if got_u != *want {
                return Err(format!(
                    "job {i} word {j}: gpu {got_u:#010x} != cpu {want:#010x}"
                ));
            }
        }
    }
    Ok(())
}
