from auto_eval.success_detector.human import HumanHandDetector
from auto_eval.success_detector.paligemma import PaligemmaDetector

detectors = {
    "paligemma": PaligemmaDetector,
    "none": HumanHandDetector,
}
