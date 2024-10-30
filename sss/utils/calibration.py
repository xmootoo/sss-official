import torch.nn.functional as F
import torch
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from scipy import optimize
import unittest
from sklearn.metrics import log_loss
from sss.config.config import Global
from sss.utils.classification import sync, gather_tensor
from venn_abers import VennAbersCalibrator

import warnings
warnings.filterwarnings("ignore")

class CalibrationModel:
    def __init__(self, rank, args):
        self.calibration_model = args.exp.calibration_model
        self.rank = rank
        self.args = args
        self.sklearn_model = self.get_window_calibrator(self.calibration_model)
        self.device = None

    # TODO: Implement channel probability aggregation for channel mode (only works for window mode for now)
    def compile_predictions(self, logits, targets, ch_ids=None):
        self.device = logits.device

        if self.args.ddp.ddp:
            sync(self.args)
            logits = gather_tensor(logits)
            targets = gather_tensor(targets)
            sync(self.args)

        if self.args.open_neuro.ch_loss_type == "BCE":
            window_probs = F.sigmoid(logits)
            self.args.data.num_classes = 2
        elif self.args.open_neuro.ch_loss_type == "CE":
            window_probs = F.softmax(logits, dim=-1)
            self.args.data.num_classes = logits.size(-1)
        else:
            raise ValueError(f"Invalid loss_type: {self.args.open_neuro.ch_loss_type}. Use 'BCE' or 'CE'.")
        return window_probs, targets, None

    def train(self, window_probs, labels, ch_ids=None):
        window_probs = window_probs.cpu().numpy()
        labels = labels.cpu().numpy()

        if self.args.exp.calibration_type == "channel":
            ch_probs = []
            ch_labels = []
            for ch_id in torch.unique(ch_ids):
                mask = (ch_ids == ch_id).cpu().numpy()
                ch_prob = np.mean(window_probs[mask])
                ch_label = labels[mask][0]
                ch_probs.append(ch_prob)
                ch_labels.append(ch_label)

            probs = np.array(ch_probs)
            labels = np.array(ch_labels)
        else:
            probs = window_probs

        if self.calibration_model in ["platt_scaling", "ensemble"]:
            probs = probs.reshape(-1, 1)

        if self.calibration_model == "venn_abers":
            self.p_cal = np.column_stack([1 - probs, probs]) if self.args.open_neuro.task=="binary" else probs
            self.y_cal = labels
            return

        if self.calibration_model == "isotonic_regression":
            if self.args.data.num_classes > 2:
                for i in range(self.args.data.num_classes):
                    self.sklearn_model[i].fit(probs[:, i], (labels == i).astype(int))
            else:
                self.sklearn_model.fit(probs, labels)
        else:
            self.sklearn_model.fit(probs, labels)

    def calibrate(self, probs):
        if isinstance(probs, torch.Tensor):
            probs_np = probs.cpu().numpy()
        elif isinstance(probs, np.ndarray):
            probs_np = probs  # Already a NumPy array, no conversion needed
        else:
            probs_np = np.array(probs)

        if self.calibration_model in ["platt_scaling", "ensemble"]:
            probs_np = probs_np.reshape(-1, 1)

        if self.calibration_model == "isotonic_regression":
            if self.args.data.num_classes > 2:
                calibrated_probs = np.column_stack([model.predict(probs_np[:, i]) for i, model in enumerate(self.sklearn_model)])
                calibrated_probs /= calibrated_probs.sum(axis=1, keepdims=True)
            else:
                calibrated_probs = self.sklearn_model.predict(probs_np)
        elif self.calibration_model == "venn_abers":
            p_test = np.column_stack([1 - probs_np, probs_np])
            calibrated_probs = self.sklearn_model.predict_proba(p_cal=self.p_cal, y_cal=self.y_cal, p_test=p_test)
        elif self.calibration_model in ["platt_scaling", "ensemble"]:
            calibrated_probs = self.sklearn_model.predict_proba(probs_np)
        else:
            calibrated_probs = self.sklearn_model.predict(probs_np)

        final_probs = torch.from_numpy(calibrated_probs).to(self.device)

        if self.args.data.num_classes > 2:
            return final_probs
        else:
            return final_probs[:, 1] if len(final_probs.size()) == 2 else final_probs

    def get_window_calibrator(self, calibration_model):
        print(f"Selected calibration model: {calibration_model}")
        if calibration_model == "isotonic_regression":
            if self.args.data.num_classes == 2:
                return IsotonicRegression(out_of_bounds='clip')
            elif self.args.data.num_classes > 2:
                return [IsotonicRegression(out_of_bounds='clip') for _ in range(self.args.data.num_classes)]
        elif calibration_model == "venn_abers":
            return VennAbersCalibrator(
                random_state=self.args.exp.seed,
                inductive=self.args.exp.va_inductive,
                n_splits=self.args.exp.va_splits)
        elif calibration_model == "platt_scaling":
            base_clf = LinearSVC(random_state=self.args.exp.seed)
            return CalibratedClassifierCV(estimator=base_clf, method='sigmoid', cv=5)
        elif calibration_model == "beta_calibration":
            return BetaCalibration()
        elif calibration_model == "ensemble":
            base_clf = RandomForestClassifier(n_estimators=100, random_state=self.args.exp.seed)
            return CalibratedClassifierCV(estimator=base_clf, method='isotonic', cv=5)
        else:
            raise NotImplementedError(f"Calibration model '{calibration_model}' not implemented.")

class BetaCalibration:
    def __init__(self):
        self.a, self.b = None, None

    def fit(self, probs, labels):
        def beta_nll(params):
            return -np.sum(labels * np.log(self._calibrate_probs(probs, params[0], params[1])) +
                           (1 - labels) * np.log(1 - self._calibrate_probs(probs, params[0], params[1])))

        result = optimize.minimize(beta_nll, [1, 1], method='Nelder-Mead')
        self.a, self.b = result.x

    def predict(self, probs):
        return self._calibrate_probs(probs, self.a, self.b)

    def _calibrate_probs(self, probs, a, b):
        return (probs ** a) / ((probs ** a) + ((1 - probs) ** b))

class TestCalibrationModel(unittest.TestCase):
    def test_calibration_methods(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        args = Global()
        n_samples = 1000
        methods = ["isotonic_regression", "platt_scaling", "beta_calibration", "ensemble", "venn_abers"]

        for method in methods:
            with self.subTest(method=method):
                logits = torch.randn(n_samples).to(device)
                labels = torch.randint(0, 2, (n_samples,))

                calibrator = CalibrationModel(method, args)

                window_probs, targets, _ = calibrator.compile_predictions(logits, labels)

                calibrator.train(window_probs, targets, args)

                new_probs = torch.rand(n_samples).to(device)

                calibrated_probs = calibrator.calibrate(new_probs)

                self.assertEqual(calibrated_probs.size(), (n_samples,))
                self.assertTrue(torch.all(calibrated_probs >= 0) and torch.all(calibrated_probs <= 1))

    def test_multiclass_calibration(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        n_samples = 1000
        n_classes = 4
        args = Global()
        args.open_neuro.ch_loss_type = "CE"
        args.data.num_classes = 4
        args.exp.va_inductive = True
        args.open_neuro.task = "multi"

        logits = torch.randn(n_samples, n_classes).to(device)
        labels = torch.randint(0, n_classes, (n_samples,))
        for method in ["isotonic_regression", "venn_abers"]:

            calibrator = CalibrationModel(method, args)

            window_probs, targets, _ = calibrator.compile_predictions(logits, labels)

            calibrator.train(window_probs, targets, args)

            new_probs = torch.randn((n_samples, n_classes)).to(device)
            new_probs = F.softmax(new_probs, dim=1)
            print(f"New probs shape: {new_probs.shape}")

            calibrated_probs = calibrator.calibrate(new_probs)

            self.assertEqual(calibrated_probs.shape, (n_samples, n_classes))
            self.assertTrue(torch.all(calibrated_probs >= 0) and torch.all(calibrated_probs <= 1))
            self.assertTrue(torch.allclose(calibrated_probs.sum(dim=1), torch.ones(n_samples).to(device), atol=1e-6))

if __name__ == '__main__':
    unittest.main()
