"""Auto-learning subpackage — mines resolved sessions into reusable
lessons that the intake runner retrieves on subsequent sessions.

Public surface
--------------
- :class:`LessonExtractor`  — pure-function distillation of one
  session's event log + final row into a :class:`SessionLessonRow`.

The store layer (:class:`LessonStore`) lives under
:mod:`runtime.storage.lesson_store`; the nightly batch refresher
(M7) lives in :mod:`runtime.learning.scheduler`.
"""
from runtime.learning.extractor import LessonExtractor, EXTRACTOR_VERSION

__all__ = ["LessonExtractor", "EXTRACTOR_VERSION"]
