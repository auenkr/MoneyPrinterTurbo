import os
import unittest

from PIL import ImageFont

from app.services import karaoke
from app.utils import utils

FONT = os.path.join(utils.font_dir(), "Anton-Regular.ttf")


def cues(*words, step=0.4):
    return [((i * step, i * step + step - 0.05), w) for i, w in enumerate(words)]


class KaraokeTest(unittest.TestCase):
    def style(self, **kw):
        return karaoke.Style(font_path=FONT, font_size=100, max_width=900, **kw)

    def test_chunks_respect_max_words_and_sentence_end(self):
        style = self.style()
        font = ImageFont.truetype(FONT, 100)
        words = [karaoke.Word(s, e, t) for (s, e), t in cues("a", "b.", "c", "d", "e", "f")]
        chunks = karaoke.chunk_words(words, style, font)
        self.assertEqual([[w.text for w in c] for c in chunks], [["a", "b."], ["c", "d", "e"], ["f"]])

    def test_clips_cover_timeline_without_gaps(self):
        clips = karaoke.build_clips(cues("one", "two", "three", "four"), self.style(), 1080, 1920)
        self.assertEqual(len(clips), 4)
        for a, b in zip(clips, clips[1:]):
            self.assertAlmostEqual(a.start + a.duration, b.start, places=3)

    def test_long_word_shrinks_to_fit(self):
        clips = karaoke.build_clips(cues("INCOMPREHENSIBILITIES"), self.style(), 1080, 1920)
        self.assertLessEqual(clips[0].w, 1080)
        self.assertGreaterEqual(clips[0].pos(0)[0], 0)

    def test_overlapping_cue_keeps_last_word(self):
        data = [((0.0, 0.5), "a"), ((0.4, 0.9), "b"), ((0.45, 1.0), "c"), ((0.3, 1.2), "d")]
        clips = karaoke.build_clips(data, self.style(), 1080, 1920)
        self.assertTrue(all(c.duration > 0 for c in clips))

    def test_empty_cues(self):
        self.assertEqual(karaoke.build_clips([], self.style(), 1080, 1920), [])

    def test_join_gap_mixed_text(self):
        self.assertEqual(karaoke._line(["你", "好", "AI", "model"]), "你好 AI model")


if __name__ == "__main__":
    unittest.main()


class DuckingTest(unittest.TestCase):
    def test_gain_low_under_speech_high_in_pauses(self):
        from app.services.video import speech_gain_envelope

        gain = speech_gain_envelope([((1.0, 2.0), "a"), ((5.0, 6.0), "b")])
        g = gain([0.0, 1.5, 3.5, 5.5, 9.0])
        self.assertAlmostEqual(g[1], 1.0)
        self.assertAlmostEqual(g[3], 1.0)
        self.assertAlmostEqual(g[0], 2.5)
        self.assertAlmostEqual(g[2], 2.5)
        self.assertAlmostEqual(g[4], 2.5)
