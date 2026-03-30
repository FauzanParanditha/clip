import unittest

from clip_factory.config import Settings
from clip_factory.contracts import RenderRequest, TrackingBox
from clip_factory.rendering import build_ffmpeg_command, build_filter_chain, estimate_focus_ratio


class RenderingTests(unittest.TestCase):
    def test_estimate_focus_ratio_prefers_detected_subject_center(self) -> None:
        boxes = [
            TrackingBox(frame_ms=0, frame_width=1920, frame_height=1080, x=1100, y=120, width=300, height=300),
            TrackingBox(frame_ms=1000, frame_width=1920, frame_height=1080, x=1000, y=140, width=280, height=280),
        ]
        ratio = estimate_focus_ratio(boxes)
        self.assertIsNotNone(ratio)
        self.assertGreater(ratio, 0.55)

    def test_build_ffmpeg_command_includes_subtitles_and_loudnorm(self) -> None:
        settings = Settings.from_env()
        request = RenderRequest(
            job_id="job_test",
            clip_id="clip_test",
            input_video_path="/tmp/source.mp4",
            output_path="/tmp/output.mp4",
            subtitle_path="/tmp/clip_test.ass",
            start_ms=12_000,
            end_ms=48_000,
            hook_text="Hook",
            keywords=["economy"],
        )
        command = build_ffmpeg_command(settings, request)
        command_text = " ".join(command)
        self.assertIn("loudnorm=I=-16:LRA=11:TP=-1.5", command_text)
        self.assertIn("/tmp/clip_test.ass", command_text)
        self.assertIn("/tmp/output.mp4", command_text)

    def test_build_filter_chain_uses_tracking_when_available(self) -> None:
        request = RenderRequest(
            job_id="job_test",
            clip_id="clip_test",
            input_video_path="/tmp/source.mp4",
            output_path="/tmp/output.mp4",
            subtitle_path="/tmp/clip_test.ass",
            start_ms=0,
            end_ms=30_000,
            hook_text="Hook",
            keywords=["economy"],
            tracking_boxes=[
                TrackingBox(frame_ms=0, frame_width=1920, frame_height=1080, x=960, y=100, width=320, height=320)
            ],
        )
        filter_chain = build_filter_chain(request)
        self.assertIn("subtitles=", filter_chain)
        self.assertIn("iw*0.", filter_chain)

    def test_build_filter_chain_uses_contain_pad_when_requested(self) -> None:
        request = RenderRequest(
            job_id="job_test",
            clip_id="clip_test",
            input_video_path="/tmp/source.mp4",
            output_path="/tmp/output.mp4",
            subtitle_path="/tmp/clip_test.ass",
            start_ms=0,
            end_ms=30_000,
            hook_text="Hook",
            keywords=["economy"],
            crop_mode="contain",
        )
        filter_chain = build_filter_chain(request)
        self.assertIn("force_original_aspect_ratio=decrease", filter_chain)
        self.assertIn("pad=1080:1920", filter_chain)
        self.assertNotIn("crop='if(gte(iw/ih,9/16)", filter_chain)


if __name__ == "__main__":
    unittest.main()
