/// This file holds the main helpers for command buffers, synchronization and the exact passing of a
/// rendering and drawing.
use ash::vk;

mod command_buffers;
pub use command_buffers::*;

// All the per frame data
pub struct FrameSlot {
    pub command_buffer: vk::CommandBuffer,
    pub in_flight_fence: vk::Fence,
    pub image_available_semaphore: vk::Semaphore,
    pub render_finished_semaphore: vk::Semaphore,
}

pub struct FrameSubmitInfo<'a> {
    command_buffer_info: [vk::CommandBufferSubmitInfo<'a>; 1],
    wait_semaphore_info: [vk::SemaphoreSubmitInfo<'a>; 1],
    signal_semaphore_info: [vk::SemaphoreSubmitInfo<'a>; 1],
}

impl<'info> FrameSubmitInfo<'info> {
    pub fn submit_infos<'submit>(&'submit self) -> [vk::SubmitInfo2<'submit>; 1]
    where
        'info: 'submit,
    {
        [vk::SubmitInfo2::default()
            .wait_semaphore_infos(&self.wait_semaphore_info)
            .command_buffer_infos(&self.command_buffer_info)
            .signal_semaphore_infos(&self.signal_semaphore_info)]
    }
}

pub fn create_submit_info(frame: &FrameSlot) -> FrameSubmitInfo<'_> {
    FrameSubmitInfo {
        command_buffer_info: [vk::CommandBufferSubmitInfo::default()
            .command_buffer(frame.command_buffer)],
        wait_semaphore_info: [vk::SemaphoreSubmitInfo::default()
            .semaphore(frame.image_available_semaphore)
            .stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)],
        signal_semaphore_info: [vk::SemaphoreSubmitInfo::default()
            .semaphore(frame.render_finished_semaphore)
            .stage_mask(vk::PipelineStageFlags2::ALL_GRAPHICS)],
    }
}
